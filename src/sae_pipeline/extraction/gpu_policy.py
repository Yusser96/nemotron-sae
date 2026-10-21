"""Pure memory policy for shared GPU activation streams."""

from dataclasses import dataclass

GiB = 1024**3


@dataclass(frozen=True)
class BufferLimits:
    slots: int
    high: int
    low: int
    frame_bytes: int
    max_bytes: int

    @classmethod
    def plan(
        cls,
        frame_bytes,
        free_bytes,
        *,
        workers=0,
        max_slots=64,
        budget_bytes=8 * GiB,
        reserve_bytes=12 * GiB,
        worker_reserve_bytes=5 * GiB,
        high_fraction=0.75,
        low_fraction=0.25,
    ):
        if not 0 < low_fraction < high_fraction < 1:
            raise ValueError("Ordered watermarks in (0,1) required")
        if frame_bytes <= 0 or max_slots < 4:
            raise ValueError("Positive frame size and at least four slots required")
        available = min(budget_bytes, free_bytes - reserve_bytes - workers * worker_reserve_bytes)
        slots = min(max_slots, available // frame_bytes)
        if slots < 4:
            raise RuntimeError("Insufficient VRAM for workers, workspace and four activation frames")
        return cls(
            slots,
            max(2, int(slots * high_fraction)),
            max(1, min(int(slots * low_fraction), int(slots * high_fraction) - 1)),
            frame_bytes,
            slots * frame_bytes,
        )

    def permits(self, outstanding):
        return outstanding < self.slots


def limits_for_device(policy, *, total_bytes, free_bytes, frame_bytes, workers):
    """Use real allocated VRAM; partition labels are documentation, not capacity."""
    return BufferLimits.plan(
        frame_bytes,
        free_bytes,
        workers=workers,
        max_slots=policy.max_frames,
        budget_bytes=min(int(policy.max_gib * GiB), int(total_bytes * policy.max_vram_fraction)),
        reserve_bytes=int(policy.workspace_reserve_gib * GiB),
        worker_reserve_bytes=int(policy.worker_reserve_gib * GiB),
        high_fraction=policy.high_water_fraction,
        low_fraction=policy.low_water_fraction,
    )
