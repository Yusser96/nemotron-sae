"""Bounded activation publication with immutable ranges and durable-consumer backpressure."""

import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import shutil
from sae_pipeline.shared_lock import file_lock


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".new")
    try:
        with temporary.open("x") as f:
            json.dump(value, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows can briefly deny a replace while a reader is closing.
                # Do not hide any cluster filesystem or permissions failure.
                if os.name != "nt" or attempt == 49:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class RollingPool:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "pool.json"

    @classmethod
    def create(
        cls,
        root,
        *,
        cap_bytes,
        consumers,
        owned_roots,
        initial_entries=(),
        source_digest="",
        frame_tokens=8192,
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        pool = cls(root)
        with file_lock(root / "pool.lock"):
            if pool.path.exists():
                raise FileExistsError("Never overwrite an existing rolling pool")
            entries = [dict(e) for e in initial_entries]
            frontier = 0
            for e in entries:
                if e["start_frame"] != frontier or e["frame_count"] <= 0:
                    raise ValueError("Noncontiguous initial activation ranges")
                frontier += e["frame_count"]
                if not Path(e["path"]).exists():
                    raise FileNotFoundError(e["path"])
            if sum(e["bytes"] for e in entries) > cap_bytes:
                raise ValueError("Initial pool exceeds cap")
            value = dict(
                version=1,
                pool_id=uuid.uuid4().hex,
                cap_bytes=cap_bytes,
                source_digest=source_digest,
                frame_tokens=frame_tokens,
                next_frame=frontier,
                reserved_frontier=frontier,
                entries=entries,
                reservations={},
                gc_pending=[],
                consumers={
                    name: dict(
                        frame=info.get("frame", 0),
                        checkpoint_dir=str(info["checkpoint_dir"]),
                        active=True,
                        done=False,
                    )
                    for name, info in consumers.items()
                },
                owned_roots=[str(Path(p).resolve()) for p in owned_roots],
                producer_state="waiting",
            )
            atomic_json(pool.path, value)
        return pool

    def read(self):
        deadline = time.monotonic() + 5.0
        while True:
            try:
                return json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _save(self, v):
        v["revision"] = v.get("revision", 0) + 1
        atomic_json(self.path, v)

    def _delete(self, v, entry):
        p = Path(entry["path"])
        if p.is_symlink():
            raise RuntimeError("Refuse to reclaim a symlink")
        resolved = p.resolve()
        if (
            entry.get("format") != "per_site_v2" and p.suffix != ".safetensors"
        ) or not any(Path(r) in resolved.parents for r in v["owned_roots"]):
            raise RuntimeError(
                "Refuse reclamation outside approved activation directories"
            )
        if entry.get("format") == "per_site_v2":
            if p.exists():
                if not p.is_dir() or any(child.is_symlink() for child in p.rglob("*")):
                    raise RuntimeError("Unsafe per-site publication directory")
                shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)

    def _gc(self, v):
        active = [
            c["frame"] for c in v["consumers"].values() if c["active"] and not c["done"]
        ]
        watermark = min(active) if active else v["next_frame"]
        reclaim = [
            e
            for e in v["entries"]
            if not e.get("pinned") and e["start_frame"] + e["frame_count"] <= watermark
        ]
        if reclaim:
            v["gc_pending"].extend(reclaim)
            v["entries"] = [e for e in v["entries"] if e not in reclaim]
            self._save(v)
        # Keep every staged deletion durable until the whole batch succeeds.
        # Saving the complete pool manifest after each reclaimed frame made a
        # large catch-up checkpoint rewrite an O(n^2) amount of metadata while
        # holding the cross-node pool lock. Other consumers then timed out
        # waiting for that lock and incorrectly marked the campaign failed.
        pending = list(v["gc_pending"])
        for e in pending:
            self._delete(v, e)
        if pending:
            # Deletion is idempotent, so a crash before this final publication
            # safely retries the remaining paths from the durable gc_pending
            # list on the next pool operation.
            v["gc_pending"] = []
            self._save(v)

    @staticmethod
    def used_bytes(v):
        return sum(e["bytes"] for e in v["entries"] + v["gc_pending"]) + sum(
            e["bytes"] for e in v["reservations"].values()
        )

    def reserve(self, start_frame, frame_count, byte_bound):
        if min(frame_count, byte_bound) <= 0:
            raise ValueError("Invalid publication reservation")
        with file_lock(self.root / "pool.lock"):
            v = self.read()
            self._gc(v)
            if start_frame != v["reserved_frontier"]:
                raise RuntimeError("Missing or repeated extraction range")
            if self.used_bytes(v) + byte_bound > v["cap_bytes"]:
                return None
            key = uuid.uuid4().hex
            v["reservations"][key] = dict(
                start_frame=start_frame, frame_count=frame_count, bytes=byte_bound
            )
            v["reserved_frontier"] += frame_count
            self._save(v)
            return key

    @staticmethod
    def entry_bytes(entry):
        path = Path(entry["path"])
        if entry.get("format") == "per_site_v2":
            if not path.is_dir() or any(child.is_symlink() for child in path.iterdir()):
                raise RuntimeError("Unsafe per-site publication")
            return sum(
                child.stat().st_size for child in path.iterdir() if child.is_file()
            )
        return path.stat().st_size

    def publish(self, reservation, entry):
        with file_lock(self.root / "pool.lock"):
            v = self.read()
            if reservation not in v["reservations"]:
                raise RuntimeError(
                    "Publication reservation is missing: possible lost shared-state update or duplicate receipt; preserve output and inspect"
                )
            r = v["reservations"][reservation]
            if (
                entry["start_frame"] != r["start_frame"]
                or entry["frame_count"] != r["frame_count"]
                or entry["start_frame"] != v["next_frame"]
            ):
                raise RuntimeError("Out-of-order or mismatched durable publication")
            if entry["bytes"] > r["bytes"] or self.entry_bytes(entry) != entry["bytes"]:
                raise RuntimeError("Serialized publication exceeds reserved bytes")
            if not entry.get("site_sha256") or not entry.get("token_ids_sha256"):
                raise ValueError("Publication requires immutable checksums")
            v["entries"].append(dict(entry))
            v["next_frame"] += entry["frame_count"]
            del v["reservations"][reservation]
            v["producer_state"] = "producing"
            self._save(v)

    def lookup(self, frame):
        v = self.read()
        for e in v["entries"]:
            if e["start_frame"] <= frame < e["start_frame"] + e["frame_count"]:
                if not Path(e["path"]).exists():
                    raise FileNotFoundError(
                        "Missing committed activation fill: " + e["path"]
                    )
                return e
        if frame < v["next_frame"]:
            raise FileNotFoundError(
                "Required activation range was reclaimed; exact replay is required"
            )
        if v["producer_state"] == "failed":
            raise RuntimeError(
                "Activation producer failed; preserve consumers and inspect"
            )
        return None

    def commit(self, consumer, stream_state_path, *, done=False):
        from sae_pipeline.sae.checkpoint import load_trainer_state, companion_state_path

        p = Path(stream_state_path)
        state = load_trainer_state(p)
        cursor = state["token_stream"]
        with file_lock(self.root / "pool.lock"):
            v = self.read()
            c = v["consumers"][consumer]
            if cursor.get("pool_id") != v["pool_id"]:
                raise ValueError("Committed cursor belongs to a different pool")
            frame = int(cursor["frame"])
            if frame < c["frame"] or frame > v["next_frame"]:
                raise ValueError("Invalid committed activation cursor")
            if (
                not state["checkpoints"]
                or frame * v["frame_tokens"] != state["tokens_processed"]
            ):
                raise ValueError("Committed cursor and training token counts disagree")
            for name in state["checkpoints"].values():
                if Path(name).name != name:
                    raise ValueError("Invalid checkpoint basename")
                weights = Path(c["checkpoint_dir"]) / name
                if not weights.is_file() or not companion_state_path(weights).is_file():
                    raise FileNotFoundError("Incomplete committed resume pair")
            c.update(
                frame=frame,
                done=bool(done),
                proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                tokens_processed=state["tokens_processed"],
                at=time.time(),
            )
            self._save(v)
            self._gc(v)

    def set_producer_state(self, state):
        with file_lock(self.root / "pool.lock"):
            v = self.read()
            v["producer_state"] = state
            self._save(v)

    def recover_reservations(self):
        """Only while holding the exclusive producer allocation lease."""
        with file_lock(self.root / "pool.lock"):
            v = self.read()
            # Unpublished files remain reserved until their owner removes them.
            for r in v["reservations"].values():
                if r.get("path") and Path(r["path"]).exists():
                    raise RuntimeError(
                        "Recover the existing durable publication before releasing its reservation"
                    )
                path = self.root / "cache" / f"range-{r['start_frame']:09d}.safetensors"
                if path.exists():
                    self._delete(v, dict(path=str(path)))
                path.with_suffix(".partial.safetensors").unlink(missing_ok=True)
            v["reservations"] = {}
            v["reserved_frontier"] = v["next_frame"]
            self._save(v)
