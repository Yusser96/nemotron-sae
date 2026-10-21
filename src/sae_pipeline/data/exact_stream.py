"""Document-shuffled token stream with exact packing and shuffle-buffer resume."""

from __future__ import annotations

import random

import torch

from sae_pipeline.data.streaming import configured_sources, _load_source_dataset, _is_validation_document


class ExactTokenStream:
    def __init__(self, cfg, tokenizer):
        self.cfg, self.tokenizer = cfg, tokenizer
        self.sources = configured_sources(cfg)
        unshuffled = cfg.model_copy(update={"shuffle_buffer": 0})
        self.datasets = [_load_source_dataset(unshuffled, source) for source in self.sources]
        self.iterators = [iter(ds) for ds in self.datasets]
        self.buffers = [[] for _ in self.sources]
        self.pending = [[] for _ in self.sources]
        self.exhausted = [False for _ in self.sources]
        self.rng = [random.Random(cfg.seed + i) for i in range(len(self.sources))]
        self.active = list(range(len(self.sources)))
        self.cursor = 0
        self.language_sequences = [0 for _ in self.sources]

    def _raw(self, index):
        source = self.sources[index]
        for example in self.iterators[index]:
            text = example.get(source.text_field)
            if text and not _is_validation_document(source, example, text, self.cfg.validation_fraction):
                return text
        self.exhausted[index] = True
        raise StopIteration

    def _document(self, index):
        buffer = self.buffers[index]
        target = max(1, self.cfg.shuffle_buffer)
        while len(buffer) < target and not self.exhausted[index]:
            try:
                buffer.append(self._raw(index))
            except StopIteration:
                break
        if not buffer:
            raise StopIteration
        chosen = self.rng[index].randrange(len(buffer))
        text = buffer[chosen]
        try:
            buffer[chosen] = self._raw(index)
        except StopIteration:
            buffer.pop(chosen)
        return text

    def _sequence(self, index):
        pending = self.pending[index]
        while len(pending) < self.cfg.seq_len:
            text = self._document(index)
            pending.extend(self.tokenizer.encode(text, add_special_tokens=False))
            pending.append(self.tokenizer.eos_token_id)
        result = pending[: self.cfg.seq_len]
        del pending[: self.cfg.seq_len]
        self.language_sequences[index] += 1
        return torch.tensor(result, dtype=torch.long)

    def next_batch(self, size):
        rows = []
        for _ in range(size):
            if not self.active:
                raise StopIteration("All sources exhausted")
            position = self.cursor % len(self.active)
            index = self.active[position]
            # Unequal/exhausted sources invalidate the balanced study instead of silently changing its mix.
            rows.append(self._sequence(index))
            self.cursor = position + 1
        return torch.stack(rows)

    def state_dict(self):
        return dict(
            version=1,
            sources=[ds.state_dict() for ds in self.datasets],
            buffers=[list(b) for b in self.buffers],
            pending=[list(p) for p in self.pending],
            exhausted=list(self.exhausted),
            rng=[r.getstate() for r in self.rng],
            active=list(self.active),
            cursor=self.cursor,
            language_sequences=list(self.language_sequences),
        )

    def load_state_dict(self, state):
        if len(state["sources"]) != len(self.sources):
            raise ValueError("Source count changed")
        for ds, saved in zip(self.datasets, state["sources"]):
            ds.load_state_dict(saved)
        self.iterators = [iter(ds) for ds in self.datasets]
        self.buffers = [list(b) for b in state["buffers"]]
        self.pending = [list(p) for p in state["pending"]]
        self.exhausted = list(state["exhausted"])
        for rng, saved in zip(self.rng, state["rng"]):
            rng.setstate(saved)
        self.active, self.cursor = list(state["active"]), int(state["cursor"])
        self.language_sequences = list(state["language_sequences"])
