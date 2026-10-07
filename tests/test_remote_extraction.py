"""Producer and trainer over a real TCPStore and a Gloo peer group (the CPU stand-in for NCCL)."""

import socket
import threading

import torch

from sae_pipeline.extraction.remote import (
    CALL_SIZE_KEY,
    PRODUCER_RANK,
    TRAINER_RANK,
    RemoteExtractor,
    open_store,
    peer_group,
    take_tokens,
)


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
