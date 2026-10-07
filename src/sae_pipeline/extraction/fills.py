"""A directory of fills shared by one producer and named consumers on other nodes.

``extraction.vllm.produce_fills`` publishes ``fill_<i>/`` directories (see
``vllm_worker.sae_save``): ``token_ids.safetensors``, one ``{slug}.safetensors``
per site and the producer's ``stream_state.pt`` after that fill. Next to them,
``fills.json`` records what produced them, and each consumer keeps the index
of the fill it is reading in ``consumers/<name>`` and marks the end of its run
with ``consumers/<name>.done``. A fill every consumer has moved past is deleted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file

from sae_pipeline.extraction.base import ActivationBatch

log = logging.getLogger(__name__)

HEADER = "fills.json"
_FILL = re.compile(r"fill_(\d{6})")


def fill_dir(root: Path, index: int) -> Path:
    return root / f"fill_{index:06d}"


def published_fills(root: Path) -> list[int]:
    """Indices of the published fills under ``root``, oldest first."""
    if not root.is_dir():
        return []
    return sorted(int(match[1]) for path in root.iterdir() if (match := _FILL.fullmatch(path.name)))


def write_header(root: Path, header: dict) -> None:
    """Record what produces the fills in ``root``; refuse a producer that differs from earlier ones."""
    path = root / HEADER
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != header:
            raise ValueError(f"{root} holds fills of {existing}, not {header}")
        return
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header), encoding="utf-8")


def _consumer_path(root: Path, name: str) -> Path:
    return root / "consumers" / name


class FillQueue:
    """The producer's side: how far behind the consumers are, and the fills they no longer need."""

    def __init__(self, root: Path, consumers: list[str], *, poll_seconds: float = 10.0) -> None:
        if not consumers:
            raise ValueError("A fill queue needs at least one consumer")
        self.root, self.consumers, self.poll_seconds = root, consumers, poll_seconds

    def position(self, name: str) -> int:
        """The fill a consumer is reading; 0 before it has started."""
        path = _consumer_path(self.root, name)
        return int(path.read_text(encoding="utf-8")) if path.exists() else 0

    def _active(self) -> list[str]:
        return [name for name in self.consumers if not _consumer_path(self.root, f"{name}.done").exists()]

    def all_done(self) -> bool:
        return not self._active()

    def release(self) -> int:
        """Delete the fills every consumer still training has moved past, keeping the newest; return how many."""
        indices = published_fills(self.root)
        active = self._active()
        if not active:
            return 0
        oldest_needed = min(self.position(name) for name in active)
        # The newest fill holds the producer's resume state.
        released = [index for index in indices[:-1] if index < oldest_needed]
        for index in released:
            shutil.rmtree(fill_dir(self.root, index))
        return len(released)

    def buffered_bytes(self) -> int:
        return sum(
            file.stat().st_size for index in published_fills(self.root) for file in fill_dir(self.root, index).iterdir()
        )

    def wait_for_room(self, max_bytes: int, *, deadline: float | None = None) -> bool:
        """Release what was read and wait until less than ``max_bytes`` of fills remain.

        A full buffer waits only for the fastest consumer: once it has read
        the oldest fill, that fill goes, and consumers still behind it skip it
        (the stream is unbounded, so they train on later tokens instead). A
        slow consumer, or one waiting for its next job, holds no one back.
        Returns False instead if every consumer is done or ``deadline``
        (``time.monotonic()``) passes first.
        """
        while True:
            self.release()
            if self.all_done():
                return False
            if self.buffered_bytes() < max_bytes:
                return True
            indices = published_fills(self.root)
            if len(indices) > 1 and indices[0] < max(self.position(name) for name in self._active()):
                shutil.rmtree(fill_dir(self.root, indices[0]))
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(self.poll_seconds)


def mark_position(root: Path, name: str, index: int) -> None:
    """Record that consumer ``name`` reads fill ``index``: the fills before it may go."""
    path = _consumer_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(str(index), encoding="utf-8")
    os.replace(temporary, path)


def mark_done(root: Path, name: str) -> None:
    path = _consumer_path(root, f"{name}.done")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class FillStream:
    """A consumer's token stream and extractor in one, read from a producer's fills.

    ``next_batch`` hands out the fills' sequences in order, across fill
    boundaries, and keeps their rows for ``slugs`` for the ``extract`` call
    that follows (``sae.train_streaming.run_streaming_training`` makes them in
    that order). The next fill is read on a thread meanwhile. Its state is the
    next sequence to hand out. Fills deleted before it reads them (on resume,
    or when it falls behind a full buffer) are skipped.
    Waiting for a fill past ``deadline`` (``time.monotonic()``), or for more
    than ``max_idle_seconds`` at once, raises ``TimeoutError``: a consumer whose
    producer is gone should give its GPU back rather than hold it idle.
    """

    def __init__(
        self,
        root: Path,
        name: str,
        slugs: list[str],
        *,
        header: dict,
        device: str,
        deadline: float | None = None,
        max_idle_seconds: float | None = None,
        poll_seconds: float = 10.0,
    ) -> None:
        self.root, self.name, self.slugs, self.device = root, name, slugs, device
        self.deadline, self.max_idle_seconds, self.poll_seconds = deadline, max_idle_seconds, poll_seconds
        self.fill, self.sequence = 0, 0
        self._loaded: tuple[int, torch.Tensor, dict[str, torch.Tensor]] | None = None
        self._next: tuple[int, Future] | None = None
        self._batch: tuple[torch.Tensor, dict[str, torch.Tensor]] | None = None
        self._reader = ThreadPoolExecutor(max_workers=1)
        self._wait(lambda: (root / HEADER).exists(), idle_limit=True)
        produced = json.loads((root / HEADER).read_text(encoding="utf-8"))
        missing = sorted(set(slugs) - set(produced["slugs"]))
        if missing or {key: produced[key] for key in header} != header:
            raise ValueError(f"{root} holds fills of {produced}; this consumer needs {header} and sites {slugs}")

    def _wait(self, ready, *, idle_limit: bool = False) -> None:
        """Poll until ``ready()``; with ``idle_limit``, give up after ``max_idle_seconds`` too."""
        started = time.monotonic()
        while not ready():
            now = time.monotonic()
            if self.deadline is not None and now >= self.deadline:
                raise TimeoutError(f"No fill arrived in {self.root} before the deadline")
            if idle_limit and self.max_idle_seconds is not None and now - started >= self.max_idle_seconds:
                raise TimeoutError(f"No fill arrived in {self.root} for {self.max_idle_seconds:.0f} s")
            time.sleep(self.poll_seconds)

    def _read(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fill = fill_dir(self.root, index)
        token_ids = load_file(fill / "token_ids.safetensors")["token_ids"]
        return token_ids, {slug: load_file(fill / f"{slug}.safetensors")["activations"] for slug in self.slugs}

    def _open(self, index: int) -> None:
        """Make fill ``index`` (or the oldest after it, if it was deleted) the loaded one."""
        wanted = index
        while True:
            # Only this wait idles the consumer: the prefetch waits while it trains.
            self._wait(lambda: any(i >= wanted for i in published_fills(self.root)), idle_limit=True)
            available = min(i for i in published_fills(self.root) if i >= wanted)
            mark_position(self.root, self.name, available)
            try:
                if self._next is not None and self._next[0] == available:
                    token_ids, rows = self._next[1].result()
                else:
                    token_ids, rows = self._read(available)
                break
            except FileNotFoundError:
                # Deleted while being read: a full buffer waits only for the fastest consumer.
                self._next, wanted = None, available + 1
        if available != index:
            log.warning("%s: fills %d-%d were deleted; continuing from fill %d", self.name, index, available - 1, available)
            self.fill, self.sequence = available, 0
        self._loaded = (available, token_ids, rows)
        self._next = (available + 1, self._reader.submit(self._read_when_published, available + 1))

    def _read_when_published(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # A later fill being published means this one was too, though it may be deleted since.
        self._wait(lambda: any(i >= index for i in published_fills(self.root)))
        return self._read(index)

    def next_batch(self, batch_size: int) -> torch.Tensor:
        token_parts, row_parts = [], {slug: [] for slug in self.slugs}
        needed = batch_size
        while needed:
            if self._loaded is None or self._loaded[0] != self.fill:
                self._open(self.fill)
            _index, token_ids, rows = self._loaded
            take = min(needed, token_ids.shape[0] - self.sequence)
            token_parts.append(token_ids[self.sequence : self.sequence + take])
            seq_len = token_ids.shape[1]
            for slug in self.slugs:
                row_parts[slug].append(rows[slug][self.sequence * seq_len : (self.sequence + take) * seq_len])
            self.sequence += take
            needed -= take
            if self.sequence == token_ids.shape[0]:
                self.fill, self.sequence = self.fill + 1, 0
        batch = torch.cat(token_parts)
        self._batch = (batch, {slug: torch.cat(parts) for slug, parts in row_parts.items()})
        return batch

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        if self._batch is None or self._batch[0] is not token_ids:
            raise RuntimeError("extract() takes the token ids of the last next_batch()")
        _token_ids, rows = self._batch
        self._batch = None
        return ActivationBatch(token_ids, {slug: value.to(self.device) for slug, value in rows.items()})

    def state_dict(self) -> dict[str, object]:
        return {"fill": self.fill, "sequence": self.sequence}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.fill, self.sequence = int(state["fill"]), int(state["sequence"])

    def close(self) -> None:
        # A prefetch may still be waiting for a fill that will never come.
        self.deadline = 0.0
        self._reader.shutdown(cancel_futures=True)
