"""Bounded CUDA-IPC consumer streams and NCCL device-buffer fanout.

No activation serialization to disk: queues carry CUDA IPC handles only.
The sending process must remain alive until every consumer releases its view.
"""

from __future__ import annotations

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
