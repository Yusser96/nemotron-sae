"""Read deterministic archive frames while publication continues independently."""

import hashlib
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from safetensors import safe_open
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.rolling import RollingPool


class RollingCacheStream:
    def __init__(
        self,
        root,
        slugs,
        device,
        *,
        idle_seconds=300,
        deadline=None,
        stop=None,
        attention_projections=None,
        verify_activation_checksums=True,
    ):
        """``verify_activation_checksums=False`` trusts the writer hashes of
        device-validated publications instead of re-hashing every read on the
        CPU; token identity is still checked on every load."""
        self.pool = RollingPool(root)
        self.verify_activation_checksums = verify_activation_checksums
        self._entries = deque()
        self.pool_id = self.pool.read()["pool_id"]
        self.slugs = list(slugs)
        self.device = device
        self.idle_seconds = idle_seconds
        self.deadline = deadline
        self.stop = stop or (lambda: False)
        self.attention_projections = attention_projections or {}
        self._device_projections = {}
        self.frame = 0
        self.loaded = None
        self.batch = None
        self.checked = set()
        self.io_seconds = self.checksum_seconds = self.transfer_enqueue_seconds = 0.0
        self.frames_loaded = self.activation_bytes_loaded = self.checksum_bytes = (
            self.tokens_delivered
        ) = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch = None

    def _load(self, entry):
        started = time.monotonic()
        path = Path(entry.get("token_ids_path", entry["path"]))
        values = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            ids = f.get_tensor("token_ids")
            if (
                hashlib.sha256(
                    memoryview(ids.contiguous().view(torch.uint8).numpy())
                ).hexdigest()
                != entry["token_ids_sha256"]
            ):
                raise RuntimeError("Token archive identity differs")
            for site in self.slugs:
                if entry.get("format") == "per_site_v2":
                    with safe_open(
                        entry["site_paths"][site], framework="pt", device="cpu"
                    ) as site_file:
                        value = site_file.get_tensor(
                            entry.get("tensor_keys", {}).get(site, site)
                        )
                else:
                    value = f.get_tensor(entry.get("tensor_keys", {}).get(site, site))
                if value.shape[0] != ids.numel():
                    raise RuntimeError("Activation rows differ from token IDs: " + site)
                device_validated = entry.get("finite_validation") == "device"
                if self.verify_activation_checksums or not device_validated:
                    digest = hashlib.sha256(
                        memoryview(value.contiguous().view(torch.uint8).numpy())
                    ).hexdigest()
                    if digest != entry["site_sha256"][site]:
                        raise RuntimeError("Activation checksum differs: " + site)
                # Bounded finite checks, including all rows before use.
                if not device_validated:
                    for chunk in value.split(8192):
                        if not torch.isfinite(chunk).all():
                            raise FloatingPointError("Nonfinite cached activations")
                values[site] = (
                    value.pin_memory()
                    if torch.device(self.device).type == "cuda"
                    else value
                )
        return entry, ids, values, time.monotonic() - started

    def _lookup(self, frame):
        """Published ranges are immutable and reclaimed only behind every durable
        cursor, so the shared manifest is re-read only past the cached snapshot."""
        while self._entries and (
            self._entries[0]["start_frame"] + self._entries[0]["frame_count"]
            <= self.frame
        ):
            self._entries.popleft()
        for entry in self._entries:
            if entry["start_frame"] > frame:
                break
            if frame < entry["start_frame"] + entry["frame_count"]:
                return entry
        state = self.pool.read()
        entry = RollingPool.find(state, frame)
        self._entries = deque(
            e
            for e in state["entries"]
            if e["start_frame"] + e["frame_count"] > self.frame
        )
        return entry

    def next_batch(self, count):
        if count != 4:
            raise ValueError("Rolling training preserves four-sequence archive frames")
        began = time.monotonic()
        while True:
            if self.stop() or (
                self.deadline is not None and time.monotonic() >= self.deadline
            ):
                raise TimeoutError("Allocation is ending")
            entry = self._lookup(self.frame)
            if entry is not None:
                break
            if time.monotonic() - began >= self.idle_seconds:
                raise TimeoutError(
                    "Waiting for rolling activations; commit and continue later"
                )
            time.sleep(1)
        if self.loaded is None or self.loaded[0]["path"] != entry["path"]:
            started = time.monotonic()
            if self.prefetch is not None and self.prefetch[0] == entry["path"]:
                loaded = self.prefetch[1].result()
                self.prefetch = None
            else:
                loaded = self._load(entry)
            self.loaded = loaded
            self.io_seconds += time.monotonic() - started
            self.checksum_seconds += loaded[3]
            self.frames_loaded += 1
            self.activation_bytes_loaded += sum(
                v.numel() * v.element_size() for v in loaded[2].values()
            )
            following = entry["start_frame"] + entry["frame_count"]
            try:
                future = self._lookup(following)
            except FileNotFoundError:
                future = None
            if future is not None:
                self.prefetch = (
                    future["path"],
                    self.executor.submit(self._load, future),
                )
        entry, ids, values, _ = self.loaded
        offset = (self.frame - entry["start_frame"]) * 4
        token_ids = ids[offset : offset + 4].long().clone()
        length = token_ids.shape[1]
        start = time.monotonic()
        activations = {}
        for site, value in values.items():
            activation = value[offset * length : (offset + 4) * length].to(
                self.device, copy=True, non_blocking=True
            )
            representation = entry.get("site_representations", {}).get(site)
            if representation == "attn_out_prelinear":
                if site not in self.attention_projections:
                    raise RuntimeError(
                        f"{site}: cached pre-projection activations need the matching o_proj matrix"
                    )
                projection = self._device_projections.get(site)
                if projection is None:
                    projection = self.attention_projections[site].to(self.device)
                    self._device_projections[site] = projection
                if activation.shape[-1] != projection.shape[-1]:
                    raise RuntimeError(
                        f"{site}: cached activation and o_proj input widths disagree"
                    )
                activation = torch.nn.functional.linear(activation, projection)
            elif representation not in (
                None,
                "attn_out",
                "resid_post",
                "mamba_out",
                "moe_out",
            ):
                raise RuntimeError(
                    f"{site}: unsupported cached activation representation {representation!r}"
                )
            activations[site] = activation
        self.transfer_enqueue_seconds += time.monotonic() - start
        self.frame += 1
        self.tokens_delivered += token_ids.numel()
        self.batch = (token_ids, activations)
        return token_ids

    def extract(self, ids):
        if self.batch is None or self.batch[0] is not ids:
            raise RuntimeError("Cached input identity mismatch")
        _, values = self.batch
        self.batch = None
        return ActivationBatch(ids, values)

    def state_dict(self):
        if (
            self.loaded is not None
            and self.frame
            >= self.loaded[0]["start_frame"] + self.loaded[0]["frame_count"]
        ):
            self.loaded = None
        return dict(pool_id=self.pool_id, frame=self.frame)

    def load_state_dict(self, state):
        if state.get("pool_id") != self.pool_id or int(state["frame"]) < 0:
            raise ValueError("Incompatible rolling cursor")
        self.frame = int(state["frame"])
        self.loaded = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.loaded = None
        self.batch = None
        self.prefetch = None
        self._device_projections.clear()
