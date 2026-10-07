"""Measure a directory as an activation buffer between a producer and a trainer.

The writer saves ``--chunks`` safetensors files, each shaped like one
extraction call (``--tokens`` bfloat16 rows for every width in ``--widths``),
and publishes each by renaming it; it waits while ``--queue`` chunks are still
unread. The reader loads every chunk as it appears, deletes it (the ``consume``
lifecycle) and copies it to ``--device``. Start both roles at once against the
same directory, on one node or on two; each prints one JSON line, and the
reader's tokens/s is the rate the buffer sustains.

Usage:
    python -m sae_pipeline.cli.bench_buffer --dir <dir> --role write --widths 2688 4096 --tokens 16384
    python -m sae_pipeline.cli.bench_buffer --dir <dir> --role read --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

POLL_SECONDS = 0.005


def _chunk(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:08d}.safetensors"


def _wait_for(ready, poll: float) -> float:
    started = time.perf_counter()
    while not ready():
        time.sleep(poll)
    return time.perf_counter() - started


def _report(role: str, seconds: float, waited: float, chunks: int, tokens: int, nbytes: int) -> dict:
    return {
        "role": role,
        "host": socket.gethostname(),
        "chunks": chunks,
        "tokens": tokens,
        "seconds": seconds,
        "waiting_s": waited,
        "tokens_per_s": tokens / seconds,
        "gb_per_s": nbytes / seconds / 1e9,
    }


def write_chunks(directory: Path, widths: list[int], tokens: int, chunks: int, queue: int) -> dict:
    """Publish ``chunks`` chunks, keeping at most ``queue`` unread; ``waiting_s`` is time blocked on the reader."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    tensors = {f"site_{i}": torch.randn(tokens, w, generator=generator).bfloat16() for i, w in enumerate(widths)}
    waited = 0.0
    started = time.perf_counter()
    for index in range(chunks):
        if index >= queue:
            waited += _wait_for(lambda: not _chunk(directory, index - queue).exists(), POLL_SECONDS)
        # The reader may be on another node: it must never see a partial file.
        partial = directory / f".{_chunk(directory, index).name}.partial"
        save_file(tensors, str(partial))
        partial.rename(_chunk(directory, index))
    nbytes = chunks * sum(t.nbytes for t in tensors.values())
    return _report("write", time.perf_counter() - started, waited, chunks, chunks * tokens, nbytes)


def read_chunks(directory: Path, chunks: int, device: str) -> dict:
    """Consume ``chunks`` chunks in order; the clock starts once the first one is there."""
    _wait_for(lambda: _chunk(directory, 0).exists(), POLL_SECONDS)
    waited, tokens, nbytes = 0.0, 0, 0
    started = time.perf_counter()
    for index in range(chunks):
        path = _chunk(directory, index)
        waited += _wait_for(path.exists, POLL_SECONDS)
        tensors = load_file(path)
        path.unlink()
        for tensor in tensors.values():
            nbytes += tensor.to(device, copy=True).nbytes
        tokens += next(iter(tensors.values())).shape[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return _report("read", time.perf_counter() - started, waited, chunks, tokens, nbytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--role", choices=["write", "read"], required=True)
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--widths", type=int, nargs="+", default=[2688], help="writer: one width per site")
    parser.add_argument("--tokens", type=int, default=16384, help="writer: rows per site in each chunk")
    parser.add_argument("--queue", type=int, default=8, help="writer: most unread chunks in the directory")
    parser.add_argument("--device", default="cpu", help="reader: where each chunk is copied")
    args = parser.parse_args()

    if args.role == "write":
        result = write_chunks(args.dir, args.widths, args.tokens, args.chunks, args.queue)
    else:
        result = read_chunks(args.dir, args.chunks, args.device)
    print(json.dumps({"dir": str(args.dir), **result}), flush=True)


if __name__ == "__main__":
    main()
