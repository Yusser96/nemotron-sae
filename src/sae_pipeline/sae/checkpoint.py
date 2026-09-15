"""Shared SAE checkpoint discovery, loading, saving, and retention.

SAE weights are portable safetensors files.  Exact training resume additionally
requires the companion ``trainer_state_step_*.pt`` file with optimizer, RNG,
dead-feature, and activation-buffer state.
"""

from __future__ import annotations

import logging
import os
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)

_WEIGHTS_RE = re.compile(r"^sae_step_(\d+)\.safetensors$")
_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A locally available checkpoint and, when present, its resume state."""

    weights_path: Path
    trainer_state_path: Path | None
    step: int
    source: str


def normalize_hf_repo_id(source: str) -> str:
    """Return a Hugging Face repo ID from either a repo ID or repository URL.

    Standard repository URLs are accepted.  URLs to ``blob``/``resolve`` files
    and ``tree`` views are deliberately rejected; select a file and revision via
    the corresponding checkpoint configuration fields instead.
    """

    value = source.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _HF_HOSTS:
            raise ValueError(f"Not a Hugging Face repository URL: {source!r}")
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) not in {1, 2}:
            raise ValueError(
                "Hugging Face source must be a repository URL, not a tree, blob, "
                f"or resolve URL: {source!r}"
            )
        value = "/".join(parts)

    parts = value.split("/")
    if len(parts) not in {1, 2} or not all(parts):
        raise ValueError(
            "Hugging Face repository must have the form 'name' or 'owner/name', "
            f"got {source!r}"
        )
    return value


def normalize_hf_source(source: str | Path) -> str:
    """Normalize a local path/repo ID/repository URL for HF ``from_pretrained`` APIs."""

    local = _local_source(source)
    return str(local) if local is not None else normalize_hf_repo_id(str(source))


def checkpoint_step(path_or_name: str | Path) -> int:
    """Extract the global step from a portable SAE checkpoint filename."""

    match = _WEIGHTS_RE.match(Path(path_or_name).name)
    if match is None:
        raise ValueError(
            f"Expected a checkpoint named sae_step_<step>.safetensors, got {path_or_name!s}"
        )
    return int(match.group(1))


def companion_state_name(path_or_name: str | Path) -> str:
    """Return the companion resume-state filename for a weights filename."""

    name = Path(path_or_name).name
    match = _WEIGHTS_RE.match(name)
    if match is None:
        raise ValueError(
            f"Expected a checkpoint named sae_step_<step>.safetensors, got {name!r}"
        )
    digits = match.group(1)
    return f"trainer_state_step_{digits}.pt"


def companion_state_path(weights_path: str | Path) -> Path:
    path = Path(weights_path)
    return path.with_name(companion_state_name(path.name))


def _local_source(source: str | Path) -> Path | None:
    path = Path(source).expanduser()
    if path.exists():
        return path
    value = str(source)
    if path.is_absolute() or value.startswith((".", "~")) or value.endswith(".safetensors"):
        raise FileNotFoundError(f"Local source does not exist: {path}")
    return None


def _sorted_weight_names(names: list[str]) -> list[str]:
    return sorted(
        (name for name in names if _WEIGHTS_RE.match(PurePosixPath(name).name)),
        key=lambda name: checkpoint_step(PurePosixPath(name).name),
    )


def _select_name(
    names: list[str],
    filename: str | None,
    require_trainer_state: bool,
) -> tuple[str, str | None]:
    available = set(names)
    if filename is not None:
        selected = filename
        if selected not in available:
            raise FileNotFoundError(f"Checkpoint {selected!r} was not found")
        checkpoint_step(PurePosixPath(selected).name)
        parent = PurePosixPath(selected).parent
        state_name = str(parent / companion_state_name(selected))
        if state_name not in available:
            if require_trainer_state:
                raise FileNotFoundError(
                    f"Resume requires companion state {state_name!r}"
                )
            state_name = None
        return selected, state_name

    candidates = _sorted_weight_names(names)
    if require_trainer_state:
        complete: list[tuple[str, str]] = []
        for candidate in candidates:
            parent = PurePosixPath(candidate).parent
            state_name = str(parent / companion_state_name(candidate))
            if state_name in available:
                complete.append((candidate, state_name))
        if not complete:
            raise FileNotFoundError("No complete SAE checkpoint pairs were found")
        return complete[-1]

    if not candidates:
        raise FileNotFoundError("No SAE checkpoints were found")
    selected = candidates[-1]
    parent = PurePosixPath(selected).parent
    state_name = str(parent / companion_state_name(selected))
    return selected, state_name if state_name in available else None


def resolve_checkpoint(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str = "main",
    require_trainer_state: bool = False,
) -> ResolvedCheckpoint:
    """Resolve a local path, HF repo ID, or HF repository URL to local files."""

    local = _local_source(source)
    if local is not None:
        if local.is_file():
            if filename is not None:
                raise ValueError("checkpoint_filename cannot be used when source is a file")
            weights_path = local
            match = _WEIGHTS_RE.match(weights_path.name)
            if match is None:
                if require_trainer_state:
                    raise ValueError(
                        "Resume requires a checkpoint named sae_step_<step>.safetensors "
                        f"with a companion trainer state file, got {weights_path.name!r}"
                    )
                return ResolvedCheckpoint(
                    weights_path=weights_path,
                    trainer_state_path=None,
                    step=-1,
                    source=str(local),
                )
            step = int(match.group(1))
            state_path = companion_state_path(weights_path)
            if not state_path.exists():
                if require_trainer_state:
                    raise FileNotFoundError(
                        f"Resume requires companion state {state_path}"
                    )
                state_path = None
            return ResolvedCheckpoint(
                weights_path=weights_path,
                trainer_state_path=state_path,
                step=step,
                source=str(local),
            )

        # Top-level only, matching prune_checkpoints(); a checkpoint directory
        # is never expected to contain nested subdirectories of checkpoints.
        relative_names = [path.name for path in local.iterdir() if path.is_file()]
        selected, state_name = _select_name(
            relative_names, filename, require_trainer_state
        )
        weights_path = local / selected
        state_path = local / state_name if state_name is not None else None
        return ResolvedCheckpoint(
            weights_path=weights_path,
            trainer_state_path=state_path,
            step=checkpoint_step(weights_path),
            source=str(local),
        )

    repo_id = normalize_hf_source(source)
    if filename is not None:
        checkpoint_step(PurePosixPath(filename).name)
        selected = filename
        state_name = str(
            PurePosixPath(filename).parent / companion_state_name(filename)
        )
        if require_trainer_state:
            repo_files = list_repo_files(repo_id=repo_id, revision=revision)
            if state_name not in repo_files:
                raise FileNotFoundError(f"Resume requires companion state {state_name!r}")
    else:
        repo_files = list_repo_files(repo_id=repo_id, revision=revision)
        selected, state_name = _select_name(
            list(repo_files), None, require_trainer_state
        )
    weights_path = Path(
        hf_hub_download(repo_id=repo_id, filename=selected, revision=revision)
    )
    state_path = None
    # Weight-only consumers should not pay to download an unused pickle file.
    if require_trainer_state and state_name is not None:
        state_path = Path(
            hf_hub_download(repo_id=repo_id, filename=state_name, revision=revision)
        )
    return ResolvedCheckpoint(
        weights_path=weights_path,
        trainer_state_path=state_path,
        step=checkpoint_step(weights_path),
        source=repo_id,
    )


def load_sae_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Load a tensor-compatible, dimension-compatible SAE state dictionary."""

    state = load_file(str(checkpoint), device="cpu")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    incompatible = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in set(state) & set(expected)
        if state[key].shape != expected[key].shape
    }
    if missing or unexpected or incompatible:
        details: list[str] = []
        if incompatible:
            details.append(f"incompatible tensor dimensions: {incompatible}")
        if missing:
            details.append(f"missing tensors: {missing}")
        if unexpected:
            details.append(f"unexpected tensors: {unexpected}")
        raise ValueError(
            f"Checkpoint {checkpoint} is not compatible with this SAE ("
            + "; ".join(details)
            + ")"
        )
    model.load_state_dict(state, strict=True)


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        saved = state["torch_cuda"]
        if len(saved) != torch.cuda.device_count():
            raise ValueError(
                "Cannot exactly resume: saved CUDA RNG state has "
                f"{len(saved)} devices, current process has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([rng.cpu() for rng in saved])


def _numpy_rng_safe_globals() -> list[Any]:
    """Globals needed to unpickle the tuple returned by ``np.random.get_state()``."""

    core = getattr(np, "_core", None) or np.core  # numpy >= 2.0 vs < 2.0 layout
    globals_: list[Any] = [
        core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    return globals_


def load_trainer_state(path: str | Path) -> dict[str, Any]:
    """Load trusted state needed for exact resume.

    ``checkpoint_source`` may point at an arbitrary Hugging Face repo, so this
    loads with ``weights_only=True`` to prevent arbitrary code execution from a
    malicious or compromised checkpoint: only tensors, plain Python
    containers/scalars, and the numpy RNG-state types this file actually
    writes are unpickled.
    """

    torch.serialization.add_safe_globals(_numpy_rng_safe_globals())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid trainer state in {path}: expected a dictionary")
    return state


def _atomic_checkpoint_paths(out_dir: Path, step: int) -> tuple[Path, Path]:
    digits = f"{step:07d}"
    return (
        out_dir / f"sae_step_{digits}.safetensors",
        out_dir / f"trainer_state_step_{digits}.pt",
    )


def save_checkpoint_pair(
    out_dir: str | Path,
    step: int,
    model_state: Mapping[str, torch.Tensor],
    trainer_state: Mapping[str, Any],
    *,
    keep_last: int = 2,
) -> ResolvedCheckpoint:
    """Atomically commit a weights/state pair, then prune older complete pairs."""

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path, state_path = _atomic_checkpoint_paths(directory, step)
    nonce = uuid.uuid4().hex
    temp_weights = directory / f".{weights_path.name}.{nonce}.tmp"
    temp_state = directory / f".{state_path.name}.{nonce}.tmp"
    backup_weights = directory / f".{weights_path.name}.{nonce}.bak"
    backup_state = directory / f".{state_path.name}.{nonce}.bak"
    weights_committed = False
    state_committed = False
    weights_backed_up = False
    state_backed_up = False
    try:
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in model_state.items()},
            str(temp_weights),
        )
        torch.save(dict(trainer_state), temp_state)
        if weights_path.exists():
            os.replace(weights_path, backup_weights)
            weights_backed_up = True
        if state_path.exists():
            os.replace(state_path, backup_state)
            state_backed_up = True
        os.replace(temp_weights, weights_path)
        weights_committed = True
        os.replace(temp_state, state_path)
        state_committed = True
    except BaseException:
        temp_weights.unlink(missing_ok=True)
        temp_state.unlink(missing_ok=True)
        if weights_committed:
            weights_path.unlink(missing_ok=True)
        if state_committed:
            state_path.unlink(missing_ok=True)
        if weights_backed_up:
            os.replace(backup_weights, weights_path)
        if state_backed_up:
            os.replace(backup_state, state_path)
        raise
    else:
        backup_weights.unlink(missing_ok=True)
        backup_state.unlink(missing_ok=True)

    prune_checkpoints(directory, keep_last=keep_last)
    return ResolvedCheckpoint(
        weights_path=weights_path,
        trainer_state_path=state_path,
        step=step,
        source=str(directory),
    )


def prune_checkpoints(out_dir: str | Path, *, keep_last: int) -> None:
    """Retain the newest N complete pairs; never remove unpaired weight files."""

    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    directory = Path(out_dir)
    complete: list[tuple[int, Path, Path]] = []
    for weights_path in directory.glob("sae_step_*.safetensors"):
        try:
            step = checkpoint_step(weights_path)
        except ValueError:
            continue
        state_path = companion_state_path(weights_path)
        if state_path.exists():
            complete.append((step, weights_path, state_path))
    complete.sort(key=lambda pair: pair[0])
    for _, weights_path, state_path in complete[:-keep_last]:
        weights_path.unlink()
        state_path.unlink()
        log.info("Removed old checkpoint pair %s and %s", weights_path, state_path)


# Clear, short aliases for callers and compatibility with earlier private helpers.
latest_checkpoint = resolve_checkpoint
