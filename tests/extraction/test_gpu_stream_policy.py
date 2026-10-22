"""Frame memory bounds and exact input-frame ownership."""

import queue
import threading

import pytest
import torch

from sae_pipeline.extraction.gpu_policy import BufferLimits, GiB
from sae_pipeline.extraction.gpu_stream import GPUFrameStream


def test_large_buffer_respects_byte_budget_and_trainer_reserve():
    frame = 8192 * 2688 * 2 * 3
    limits = BufferLimits.plan(frame, 80 * GiB, workers=5)
    assert limits.slots == 64 and limits.high == 48 and limits.low == 16
    assert limits.max_bytes <= 8 * GiB
    larger = BufferLimits.plan(frame * 3, 80 * GiB, workers=5)
    assert larger.slots < 64 and larger.max_bytes <= 8 * GiB
    assert not larger.permits(larger.slots)
    assert larger.permits(larger.slots - 1)
    with pytest.raises(RuntimeError, match="Insufficient VRAM"):
        BufferLimits.plan(frame, 16 * GiB, workers=1)


def test_frame_release_waits_only_own_stream_and_acknowledges_once(monkeypatch):
    inbox, acks = queue.Queue(), queue.Queue()
    stop = threading.Event()
    ids = torch.arange(16).reshape(4, 4)
    rows = torch.arange(16 * 3).reshape(16, 3)
    inbox.put((0, ids, {"L2": rows}))
    inbox.put((1, ids + 16, {"L2": rows + 100}))
    events = []

    class Event:
        def record(self, stream):
            events.append(("record", stream))

        def synchronize(self):
            events.append(("wait",))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "own-stream")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("device-wide fence"))
    stream = GPUFrameStream(inbox, acks, stop, "warm-L2", ["L2"], "cpu")
    batch = stream.next_batch(2)
    assert torch.equal(stream.extract(batch).activations["L2"], rows[:8])
    assert acks.empty()
    batch = stream.next_batch(2)
    stream.extract(batch)
    assert acks.empty()
    assert stream.state_dict() == dict(fill=1, sequence=0)
    stream.next_batch(2)
    assert acks.get_nowait() == ("warm-L2", 0)
    assert events == [("record", "own-stream"), ("wait",)]
    stream.close()
    stream.close()
    assert acks.get_nowait() == ("warm-L2", 1) and acks.empty()


def test_missing_frame_fails_instead_of_skipping():
    inbox = queue.Queue()
    ids = torch.arange(8).reshape(2, 4)
    inbox.put((3, ids, {"L2": torch.zeros(8, 3)}))
    stream = GPUFrameStream(
        inbox,
        queue.Queue(),
        threading.Event(),
        "scratch-L2",
        ["L2"],
        "cpu",
        cursor=dict(fill=2, sequence=0),
    )
    with pytest.raises(RuntimeError, match="expected immutable frame 2"):
        stream.next_batch(2)


def test_configurable_limits_adapt_to_real_vram():
    from sae_pipeline.config import GPUBufferCfg, StreamCfg
    from sae_pipeline.extraction.gpu_policy import limits_for_device

    cfg = StreamCfg.model_validate(
        {"gpu_buffer": {"max_frames": 512, "max_gib": 32, "max_vram_fraction": 0.2}}
    )
    frame = 8192 * 2688 * 2 * 7
    h100 = limits_for_device(
        cfg.gpu_buffer,
        total_bytes=80 * GiB,
        free_bytes=79 * GiB,
        frame_bytes=frame,
        workers=7,
    )
    nvl = limits_for_device(
        cfg.gpu_buffer,
        total_bytes=94 * GiB,
        free_bytes=93 * GiB,
        frame_bytes=frame,
        workers=7,
    )
    assert h100.max_bytes <= 16 * GiB and nvl.max_bytes <= int(18.8 * GiB)
    assert nvl.slots > h100.slots
    constrained = limits_for_device(
        cfg.gpu_buffer,
        total_bytes=94 * GiB,
        free_bytes=49 * GiB,
        frame_bytes=frame,
        workers=7,
    )
    assert constrained.max_bytes <= 2 * GiB
    for invalid in (
        {"max_frames": 3},
        {"max_gib": float("inf")},
        {"max_vram_fraction": 1.1},
        {"low_water_fraction": 0.9},
        {"max_frames": 32.5},
    ):
        with pytest.raises(ValueError):
            GPUBufferCfg.model_validate(invalid)
