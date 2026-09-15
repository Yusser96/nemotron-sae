"""Streaming text loader and packed-sequence iterator.

Used identically by dev (n_documents=100) and prod (total_tokens=200B). The same code
path serves both — only the budget knob differs — so any data-induced bug shows up in dev.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from sae_pipeline.config import DataCfg, DataSourceCfg

log = logging.getLogger(__name__)


def configured_sources(cfg: DataCfg) -> list[DataSourceCfg]:
    """Return explicit multilingual sources or the backwards-compatible source."""
    if cfg.sources is not None:
        return cfg.sources
    return [DataSourceCfg(source=cfg.source, name=cfg.name, text_field=cfg.text_field)]


def _is_validation_document(source: DataSourceCfg, example: dict[str, Any], text: str, fraction: float) -> bool:
    """Assign a whole document to a stable partition without storing an index."""
    identifier = example.get("id") or example.get("uuid") or text
    digest = hashlib.sha256(f"{source.source}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def stream_documents(
    cfg: DataCfg,
    source: DataSourceCfg | None = None,
    partition: str = "train",
) -> Iterator[str]:
    """Yield one deterministic train or validation document stream."""
    source = source if source is not None else configured_sources(cfg)[0]
    if partition not in {"train", "validation"}:
        raise ValueError(f"Unknown partition {partition!r}")
    docs = _stream_source_documents(cfg, source, partition)
    n_emitted = 0
    for text in docs:
        yield text
        n_emitted += 1
        if cfg.n_documents is not None and n_emitted >= cfg.n_documents:
            return
    if cfg.n_documents is not None and n_emitted < cfg.n_documents:
        raise RuntimeError(
            f"Source {source.source!r} ended after {n_emitted} documents; "
            f"requested {cfg.n_documents}"
        )


def _stream_source_documents(
    cfg: DataCfg,
    source: DataSourceCfg,
    partition: str,
) -> Iterator[str]:
    """Yield all valid documents from one source without applying a budget."""
    log.info(
        "Streaming dataset %s name=%s split=%s streaming=%s",
        source.source, source.name, cfg.split, cfg.streaming,
    )
    load_kwargs: dict[str, Any] = {"split": cfg.split, "streaming": cfg.streaming}
    if source.name is not None:
        load_kwargs["name"] = source.name
    ds = load_dataset(source.source, **load_kwargs)

    if cfg.streaming and cfg.shuffle_buffer:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    for ex in ds:
        text = ex.get(source.text_field)
        if not text:
            continue
        is_validation = _is_validation_document(source, ex, text, cfg.validation_fraction)
        if (partition == "validation") != is_validation:
            continue
        yield text


def _pack_token_ids(
    token_chunks: Iterator[list[int]],
    seq_len: int,
    total_tokens: int | None = None,
) -> Iterator[torch.Tensor]:
    """Pack already-tokenized document chunks into fixed-length sequences."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if total_tokens is not None and total_tokens <= 0:
        raise ValueError("total_tokens must be positive")

    pending: list[int] = []
    n_yielded_tokens = 0
    for ids in token_chunks:
        pending.extend(ids)
        while len(pending) >= seq_len:
            seq = pending[:seq_len]
            pending = pending[seq_len:]
            yield torch.tensor(seq, dtype=torch.long)
            n_yielded_tokens += seq_len
            if total_tokens is not None and n_yielded_tokens >= total_tokens:
                return

    if total_tokens is not None and n_yielded_tokens < total_tokens:
        raise RuntimeError(
            f"Document stream ended after {n_yielded_tokens} packed tokens; "
            f"requested {total_tokens}"
        )


def pack_token_stream(
    docs: Iterator[str],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    eos_separator: bool = True,
    total_tokens: int | None = None,
) -> Iterator[torch.Tensor]:
    """Tokenize a stream of documents and yield (seq_len,) int64 tensors.

    Documents are concatenated with the EOS token between them, then chopped into
    fixed-length sequences. Final partial sequence is dropped.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS — required for packing.")

    def tokenized() -> Iterator[list[int]]:
        for text in docs:
            ids = list(tokenizer.encode(text, add_special_tokens=False))
            if eos_separator:
                ids.append(eos_id)
            yield ids

    yield from _pack_token_ids(tokenized(), seq_len, total_tokens)


def batch_sequences(
    seqs: Iterator[torch.Tensor], batch_size: int
) -> Iterator[torch.Tensor]:
    """Stack `batch_size` sequences into a (B, T) int64 batch."""
    chunk: list[torch.Tensor] = []
    for s in seqs:
        chunk.append(s)
        if len(chunk) == batch_size:
            yield torch.stack(chunk, dim=0)
            chunk = []
    if chunk:
        yield torch.stack(chunk, dim=0)


def make_token_loader(
    cfg: DataCfg,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    partition: str = "train",
    languages: set[str] | None = None,
) -> Iterator[torch.Tensor]:
    """Compose deterministic partitioned streams into equally weighted batches."""
    sources = configured_sources(cfg)
    if languages is not None:
        sources = [s for s in sources if s.language in languages]
    if not sources:
        raise ValueError("No configured data source matches the requested language")

    if partition == "train":
        total_tokens = cfg.total_tokens
    elif partition == "validation":
        total_tokens = cfg.validation_tokens_per_language
    else:
        raise ValueError(f"Unknown partition {partition!r}")
    if total_tokens is None and cfg.n_documents is None:
        raise ValueError(f"No token budget configured for {partition} partition")
    if total_tokens is not None and total_tokens % (cfg.seq_len * len(sources)):
        raise ValueError("Token budget must divide exactly into full, equally weighted sequences")

    if cfg.n_documents is not None and len(sources) > 1:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer has no EOS — required for packing.")

        def balanced_tokens() -> Iterator[list[int]]:
            streams = [iter(_stream_source_documents(cfg, source, partition)) for source in sources]
            token_counts = [0] * len(streams)
            active = list(range(len(streams)))
            documents_seen = 0
            while documents_seen < cfg.n_documents:
                if not active:
                    raise RuntimeError(
                        f"All sources ended after {documents_seen} documents; "
                        f"requested {cfg.n_documents}"
                    )
                source_index = min(active, key=lambda index: (token_counts[index], index))
                try:
                    text = next(streams[source_index])
                except StopIteration:
                    active.remove(source_index)
                    continue
                ids = list(tokenizer.encode(text, add_special_tokens=False))
                ids.append(eos_id)
                token_counts[source_index] += len(ids)
                documents_seen += 1
                yield ids

        seqs = _pack_token_ids(balanced_tokens(), cfg.seq_len)
    elif cfg.n_documents is not None:
        seqs = pack_token_stream(
            stream_documents(cfg, sources[0], partition=partition), tokenizer,
            seq_len=cfg.seq_len,
        )
    else:
        per_source_tokens = total_tokens // len(sources)  # type: ignore[operator]
        streams = [
            pack_token_stream(
                _stream_source_documents(cfg, source, partition), tokenizer,
                seq_len=cfg.seq_len, total_tokens=per_source_tokens,
            )
            for source in sources
        ]

        def interleaved() -> Iterator[torch.Tensor]:
            active = list(streams)
            while active:
                remaining: list[Iterator[torch.Tensor]] = []
                for stream in active:
                    try:
                        yield next(stream)
                    except StopIteration:
                        continue
                    remaining.append(stream)
                active = remaining

        seqs = interleaved()
    yield from batch_sequences(seqs, batch_size=batch_size)
