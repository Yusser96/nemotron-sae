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
from pathlib import Path

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

    def all_done(self) -> bool:
        return all(_consumer_path(self.root, f"{name}.done").exists() for name in self.consumers)

    def release(self) -> int:
        """Delete the fills every consumer has moved past, keeping the newest; return how many."""
        indices = published_fills(self.root)
        oldest_needed = min(self.position(name) for name in self.consumers)
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

        Returns False instead if every consumer is done or ``deadline``
        (``time.monotonic()``) passes first.
        """
        while True:
            self.release()
            if self.all_done():
                return False
            if self.buffered_bytes() < max_bytes:
                return True
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
