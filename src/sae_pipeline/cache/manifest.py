"""JSON manifest for an activation cache.

One manifest per (run_id, layer, component) cache directory. It enumerates the
shards that belong to the cache so readers don't have to glob.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CacheManifest:
    run_id: str
    model: str
    dtype: str               # "bfloat16" | "float32" etc.
    layer: int
    component: str           # e.g. "resid_post" or "expert.42"
    d_activation: int
    n_shards: int = 0
    total_tokens: int = 0
    shard_paths: list[str] = field(default_factory=list)
    shuffle_seed: int = 42
    dataset_fingerprint: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def read(cls, path: str | Path) -> "CacheManifest":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def manifest_path_for(cache_dir: Path | str, run_id: str, slug: str) -> Path:
    return Path(cache_dir) / run_id / slug / "manifest.json"


def shards_dir_for(cache_dir: Path | str, run_id: str, slug: str) -> Path:
    return Path(cache_dir) / run_id / slug
