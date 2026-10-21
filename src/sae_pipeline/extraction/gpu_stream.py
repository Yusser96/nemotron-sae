"""Bounded CUDA-IPC consumer streams and NCCL device-buffer fanout.

No activation serialization to disk: queues carry CUDA IPC handles only.
The sending process must remain alive until every consumer releases its view.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sae_pipeline.extraction.base import ActivationBatch


class GPUFrameStream:
    def __init__(
        self, inbox, acknowledgments, stopped, name, slugs, device, cursor=None, root=None, **kwargs
    ):
        self.inbox, self.acks, self.stopped = inbox, acknowledgments, stopped
        self.name, self.slugs, self.device = name, slugs, device
        self.fill, self.sequence = 0, 0
        if cursor:
            self.load_state_dict(cursor)
        self.loaded = None
        self.batch = None
        self.wait_seconds = 0.0
        self.frames = 0
        self.root = Path(root) if root else None

    def _release(self):
        if self.loaded is not None:
            index = self.loaded[0]
            # All training on the frame must finish before its IPC storage is reused.
            # Wait only for this trainer's stream, allowing unrelated GPU work to overlap.
            completed = torch.cuda.Event()
            completed.record(torch.cuda.current_stream(torch.device(self.device)))
            completed.synchronize()
            self.loaded = None
            self.batch = None
            self.acks.put((self.name, index))

    def next_batch(self, count):
        if self.loaded is not None and self.loaded[0] != self.fill:
            self._release()
        if self.loaded is None:
            started = time.monotonic()
            while True:
                if self.stopped.is_set():
                    raise TimeoutError("Device stream paused by coordinator")
                try:
                    item = self.inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    raise StopIteration
                index, tokens, rows = item
                if index != self.fill:
                    raise RuntimeError(f"{self.name}: expected immutable frame {self.fill}, received {index}")
                self.loaded = (index, tokens, rows)
                self.frames += 1
                break
            self.wait_seconds += time.monotonic() - started
        index, tokens, rows = self.loaded
        end = self.sequence + count
        if end > tokens.shape[0]:
            raise ValueError("A GPU minibatch cannot cross its immutable extraction frame")
        ids = tokens[self.sequence : end]
        length = tokens.shape[1]
        selected = {slug: rows[slug][self.sequence * length : end * length] for slug in self.slugs}
        self.sequence = end
        if end == tokens.shape[0]:
            self.fill += 1
            self.sequence = 0
        self.batch = (ids, selected)
        return ids

    def extract(self, ids):
        if self.batch is None or self.batch[0] is not ids:
            raise RuntimeError("Mismatched device frame")
        _, rows = self.batch
        self.batch = None
        return ActivationBatch(ids, rows)

    def state_dict(self):
        return {"fill": self.fill, "sequence": self.sequence}

    def load_state_dict(self, state):
        self.fill, self.sequence = int(state["fill"]), int(state["sequence"])

    def close(self):
        self._release()


def nccl_group(store, rank, size, device):
    torch.cuda.set_device(device)
    return dist.ProcessGroupNCCL(dist.PrefixStore("gpu-activations", store), rank, size)


def broker(rank, store, group, jobs, inboxes, acks, stopped, slots=64, byte_budget=None):
    """Receive once per GPU, publish only complete frames, retain until all ACKs.

    `released` includes empty frames and out-of-order completions. The producer
    subtracts it from its own send count so in-flight frames also consume credit.
    """
    pending = {}
    next_frame = 0
    released = 0
    pending_bytes = 0

    def release(index):
        nonlocal released, pending_bytes
        values, _ = pending.pop(index)
        pending_bytes -= sum(v.numel() * v.element_size() for v in values.values())
        released += 1

    def reclaim():
        nonlocal released
        while True:
            try:
                name, index = acks.get_nowait()
            except queue.Empty:
                break
            if index in pending:
                pending[index][1].discard(name)
        finished = {
            j["id"] for j in jobs if store.check(["done/" + j["id"]]) or store.check(["paused/" + j["id"]])
        }
        for index in list(pending):
            pending[index][1].difference_update(finished)
            if not pending[index][1]:
                release(index)
        store.set(f"released/{rank}", str(released))
        store.set(f"credit/{rank}", str(slots - len(pending)))
        store.set(f"buffer/{rank}", json.dumps(dict(frames=len(pending), bytes=pending_bytes, slots=slots)))

    reclaim()
    while not stopped.is_set():
        reclaim()
        if len(pending) >= slots:
            time.sleep(0.01)
            continue
        key = f"frame/{rank}/{next_frame}"
        if not store.check([key]):
            time.sleep(0.01)
            continue
        meta = json.loads(store.get(key))
        store.delete_key(key)
        next_frame += 1
        if meta.get("stop"):
            break
        index = meta["index"]
        tokens = torch.tensor(meta["tokens"], dtype=torch.long)
        size = tokens.numel() * sum(meta["widths"].values()) * 2
        if byte_budget is not None and pending_bytes + size > byte_budget:
            raise RuntimeError("GPU activation byte budget would be exceeded")
        values = {}
        receives = []
        for slug, width in meta["widths"].items():
            value = torch.empty(tokens.numel(), width, dtype=torch.bfloat16, device="cuda:" + str(rank))
            receives.append(group.recv([value], 0, 0))
            values[slug] = value
        for work in receives:
            work.wait()
        # CUDA IPC serialization happens in a Queue feeder thread. Make this
        # receive stream ready before handing tensors to it; no device-wide fence.
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()
        needed = {
            j["id"]
            for j in jobs
            if index >= j["cursor"]["fill"]
            and not store.check(["done/" + j["id"]])
            and not store.check(["paused/" + j["id"]])
        }
        if needed:
            pending[index] = (values, needed)
            pending_bytes += size
            for job in jobs:
                if job["id"] in needed:
                    inboxes[job["id"]].put((index, tokens, {s: values[s] for s in job["slugs"]}))
        else:
            released += 1
        reclaim()
    for job in jobs:
        inboxes[job["id"]].put(None)
    # Keep the sender alive while consumers drain queued IPC handles.
    while not store.check(["workers_closed"]):
        reclaim()
        time.sleep(0.02)


def await_helper_permission(store, stopped):
    """Grant at most one helper operation; future grants can be revoked.

    Once extraction resumes, an already-running operation may finish under
    MPS, while the helper cannot acquire its next operation until a pause.
    """
    while not stopped.is_set():
        if store.compare_set("helper/permit", "grant", "running") == b"running":
            return
        time.sleep(0.01)
    return  # Allow the already-requested operation to reach a checkpoint safely.
