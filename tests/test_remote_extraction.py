"""Producer and trainer over a real TCPStore and a Gloo peer group (the CPU stand-in for NCCL)."""

import json
import socket
import threading

import pytest
import torch

from sae_pipeline.extraction.remote import (
    CALL_SIZE_KEY,
    PRODUCER_RANK,
    SITES_KEY,
    TRAINER_RANK,
    RemoteExtractor,
    open_store,
    peer_group,
    take_tokens,
)
from sae_pipeline.extraction.vllm import serve_activations
from tests.test_vllm_extractor import SPECS, _cfg, _FakeEngine, _reference
from tests.test_vllm_worker_extension import D, _Worker


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_token_batches_go_out_and_each_calls_rows_come_back_in_site_order():
    port = _free_port()
    seen: list[torch.Tensor] = []

    def stub_producer():
        store = open_store("127.0.0.1", port, trainer=False)
        store.set(CALL_SIZE_KEY, "2")
        group = peer_group(store, PRODUCER_RANK, torch.device("cpu"))
        index = 0
        while (token_ids := take_tokens(store, index)) is not None:
            seen.append(token_ids)
            for chunk in token_ids.split(2):
                for site in (1.0, 2.0):  # each row is (site, token id)
                    rows = torch.stack([torch.full_like(chunk, site), chunk], dim=-1).reshape(-1, 2)
                    group.send([rows.float()], TRAINER_RANK, 0).wait()
            index += 1

    trainer = RemoteExtractor(host="127.0.0.1", port=port, widths={"a": 2, "b": 2}, dtype=torch.float32, device="cpu")
    producer = threading.Thread(target=stub_producer)
    producer.start()
    token_ids = torch.arange(15).reshape(5, 3)

    batch = trainer.extract(token_ids)
    trainer.close()
    producer.join(timeout=60)

    assert not producer.is_alive() and torch.equal(seen[0], token_ids)
    torch.testing.assert_close(batch.activations["a"][:, 0], torch.ones(15))
    torch.testing.assert_close(batch.activations["b"][:, 0], torch.full((15,), 2.0))
    torch.testing.assert_close(batch.activations["b"][:, 1], token_ids.flatten().float())


def test_the_trainer_receives_every_sites_rows_from_the_producer(tmp_path):
    port = _free_port()
    engines: list[_FakeEngine] = []
    served: list[int] = []

    def engine(**kwargs):
        engines.append(_FakeEngine(**kwargs))
        return engines[-1]

    producer = threading.Thread(
        target=lambda: served.append(
            serve_activations(
                _cfg(tmp_path),
                SPECS,
                host="127.0.0.1",
                port=port,
                engine_factory=engine,
                sampling_params_factory=lambda: "one-token",
            )
        )
    )
    trainer = RemoteExtractor(
        host="127.0.0.1", port=port, widths={slug: D for slug in SPECS}, dtype=torch.float32, device="cpu"
    )
    producer.start()
    batches = [torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), torch.tensor([[9, 8, 7], [6, 5, 4]])]

    received = [trainer.extract(token_ids) for token_ids in batches]
    trainer.close()
    producer.join(timeout=60)

    assert served == [2]
    assert engines[0].kwargs["max_num_batched_tokens"] == 2 * 4
    assert engines[0].calls == [2, 1, 2]
    for token_ids, batch in zip(batches, received):
        expected = _reference(engines[0].model_runner.model, token_ids)
        for slug in SPECS:
            torch.testing.assert_close(batch.activations[slug], expected[slug])


def test_an_auto_sized_producer_probes_one_step_then_answers_in_calls_that_fit(tmp_path, monkeypatch):
    # Per sequence: 3 rows of three float32 sites of width D plus int64 ids and positions,
    # and one site's assembled copy.
    per_sequence = 3 * (3 * D * 4 + 2 * 8 + D * 4)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * per_sequence + 1, 10**9))
    cfg = _cfg(tmp_path).with_overrides(
        **{
            "extraction.vllm.sequences_per_call": "auto",
            "extraction.vllm.max_sequences_per_step": 4,
            "extraction.vllm.vram_headroom_gib": 0,
        }
    )
    port = _free_port()
    engines: list[_FakeEngine] = []

    def engine(**kwargs):
        engines.append(_FakeEngine(**kwargs))
        return engines[-1]

    producer = threading.Thread(
        target=lambda: serve_activations(
            cfg, SPECS, host="127.0.0.1", port=port, engine_factory=engine, sampling_params_factory=lambda: "one-token"
        )
    )
    trainer = RemoteExtractor(
        host="127.0.0.1", port=port, widths={slug: D for slug in SPECS}, dtype=torch.float32, device="cpu"
    )
    producer.start()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    batch = trainer.extract(token_ids)
    trainer.close()
    producer.join(timeout=60)

    assert engines[0].kwargs["max_num_batched_tokens"] == 4 * 4
    assert engines[0].calls == [4, 2, 1]
    expected = _reference(engines[0].model_runner.model, token_ids)
    for slug in SPECS:
        torch.testing.assert_close(batch.activations[slug], expected[slug])


def _trainer_store(port: int, slugs: list[str], widths: list[int]):
    store = open_store("127.0.0.1", port, trainer=True)
    store.set(SITES_KEY, json.dumps({"slugs": slugs, "widths": widths, "dtype": "torch.float32"}))
    return store


def test_a_producer_hooking_other_sites_refuses_to_join():
    port = _free_port()
    _store = _trainer_store(port, ["L0_resid_post"], [D])
    worker = _Worker()
    worker.sae_install_hooks([("L0_mamba_out", "mamba_out", 0)], None)

    with pytest.raises(ValueError, match="expects sites"):
        worker.sae_join("127.0.0.1", port)


def test_rows_of_another_width_are_not_sent():
    port = _free_port()
    store = _trainer_store(port, ["L0_resid_post"], [D + 1])
    trainer_side = threading.Thread(target=lambda: peer_group(store, TRAINER_RANK, torch.device("cpu")))
    trainer_side.start()
    worker = _Worker()
    worker.sae_install_hooks([("L0_resid_post", "resid_post", 0)], None)
    worker.sae_join("127.0.0.1", port)
    trainer_side.join(timeout=60)
    with torch.no_grad():
        worker.model_runner.model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="trainer expects torch.float32 width 5"):
        worker.sae_send([[1, 2]])
