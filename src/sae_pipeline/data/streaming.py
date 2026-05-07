"""Streaming text loader and packed-sequence iterator.

Used identically by dev (n_documents=100) and prod (total_tokens=200B). The same code
path serves both — only the budget knob differs — so any data-induced bug shows up in dev.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from sae_pipeline.config import DataCfg

log = logging.getLogger(__name__)


def stream_documents(cfg: DataCfg) -> Iterator[str]:
    """Yield raw document strings from `cfg.source`. Honors `n_documents` if set."""
    log.info(
        "Streaming dataset %s name=%s split=%s streaming=%s",
        cfg.source, cfg.name, cfg.split, cfg.streaming,
    )
    load_kwargs: dict[str, Any] = {"split": cfg.split, "streaming": cfg.streaming}
    if cfg.name is not None:
        load_kwargs["name"] = cfg.name
    ds = load_dataset(cfg.source, **load_kwargs)

    if cfg.streaming and cfg.shuffle_buffer:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    n_emitted = 0
    for ex in ds:
        text = ex.get(cfg.text_field)
        if not text:
            continue
        yield text
        n_emitted += 1
        if cfg.n_documents is not None and n_emitted >= cfg.n_documents:
            return


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

    pending: list[int] = []
    n_yielded_tokens = 0

    for text in docs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pending.extend(ids)
        if eos_separator:
            pending.append(eos_id)

        while len(pending) >= seq_len:
            seq = pending[:seq_len]
            pending = pending[seq_len:]
            yield torch.tensor(seq, dtype=torch.long)
            n_yielded_tokens += seq_len
            if total_tokens is not None and n_yielded_tokens >= total_tokens:
                return


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
) -> Iterator[torch.Tensor]:
    """Compose the full pipeline: stream → tokenize → pack → batch."""
    docs = stream_documents(cfg)
    seqs = pack_token_stream(
        docs, tokenizer, seq_len=cfg.seq_len, total_tokens=cfg.total_tokens
    )
    yield from batch_sequences(seqs, batch_size=batch_size)
