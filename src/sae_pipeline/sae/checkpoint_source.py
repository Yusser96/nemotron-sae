"""Resolve local checkpoint pairs and pinned repository sources."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download, list_repo_files

_WEIGHTS_RE = re.compile(r"^sae_step_(\d+)(?:_snapshot_(\d+))?\.safetensors$")
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
    return name.replace("sae_step_", "trainer_state_step_", 1).removesuffix(".safetensors") + ".pt"


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
        key=lambda name: (checkpoint_step(PurePosixPath(name).name), _WEIGHTS_RE.match(PurePosixPath(name).name).group(2) or ""),
    )


def _select_name(names: list[str], require_trainer_state: bool) -> tuple[str, str | None]:
    """The latest checkpoint in ``names`` and its trainer state, if present or required."""
    available = set(names)
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

        if filename is not None:
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkpoint_filename must be relative to the source directory")
            # Explicit site templates may name nested raw exports. Reuse the
            # direct-file rules, including arbitrary names and resume guards.
            resolved = resolve_checkpoint(
                local / relative, require_trainer_state=require_trainer_state
            )
            return ResolvedCheckpoint(
                weights_path=resolved.weights_path,
                trainer_state_path=resolved.trainer_state_path,
                step=resolved.step,
                source=str(local),
            )

        # Automatic latest-checkpoint discovery remains top-level only,
        # matching prune_checkpoints() and avoiding unrelated nested runs.
        relative_names = [path.name for path in local.iterdir() if path.is_file()]
        selected, state_name = _select_name(relative_names, require_trainer_state)
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
        selected = filename
        basename = PurePosixPath(filename).name
        if _WEIGHTS_RE.match(basename):
            state_name = str(
                PurePosixPath(filename).parent / companion_state_name(filename)
            )
        else:
            state_name = None
        if require_trainer_state:
            if state_name is None:
                raise ValueError(
                    "Resume requires a checkpoint named sae_step_<step>.safetensors"
                )
            repo_files = list_repo_files(repo_id=repo_id, revision=revision)
            if state_name not in repo_files:
                raise FileNotFoundError(f"Resume requires companion state {state_name!r}")
    else:
        repo_files = list_repo_files(repo_id=repo_id, revision=revision)
        selected, state_name = _select_name(list(repo_files), require_trainer_state)
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
        step=checkpoint_step(weights_path) if _WEIGHTS_RE.match(weights_path.name) else -1,
        source=repo_id,
    )
