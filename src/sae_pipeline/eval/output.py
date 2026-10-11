"""Shared paths for evaluation artifacts and their plots."""
from pathlib import Path


def evaluation_directory(root, run_id, slug, partition="train", language=None, tag=None):
    if partition not in {"train", "validation"}:
        raise ValueError("Unknown evaluation partition")
    if (partition == "validation") != bool(language):
        raise ValueError("A language is required only for validation evaluations")
    path = Path(root) / run_id / slug / partition
    if language:
        path /= language
    if tag:
        path /= tag
    return path
