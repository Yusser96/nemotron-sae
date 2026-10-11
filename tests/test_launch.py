import threading
import time
from types import SimpleNamespace

import pytest

from sae_pipeline.cli.launch import run_local


def _job(module, name):
    return ["python", "-m", f"sae_pipeline.cli.{module}", name]


def test_local_jobs_obey_dependencies_and_exclusively_lease_gpus(monkeypatch):
    lock = threading.Lock()
    busy, completed = set(), set()

    def run(cmd, *, env, check):
        module, name = cmd[2].rsplit(".", 1)[-1], cmd[3]
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with lock:
            assert gpu not in busy
            busy.add(gpu)
            if module == "train_sae":
                assert {"cache_a", "cache_b"} <= completed
            if module == "evaluate":
                assert {"train_a", "train_b"} <= completed
        time.sleep(0.02 if name.endswith("a") else 0.001)
        with lock:
            busy.remove(gpu)
            completed.add(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("sae_pipeline.cli.launch.subprocess.run", run)
    jobs = [_job(module, f"{name}_{site}") for module, name in
            (("cache_activations", "cache"), ("train_sae", "train"), ("evaluate", "eval"))
            for site in ("a", "b")]
    assert run_local(jobs, ["0", "1"]) == 0
    assert len(completed) == 6


def test_failed_cache_prevents_training(monkeypatch):
    seen = []
    monkeypatch.setattr("sae_pipeline.cli.launch.subprocess.run",
                        lambda cmd, **kw: (seen.append(cmd), SimpleNamespace(returncode=1))[1])
    assert run_local([_job("cache_activations", "cache"), _job("train_sae", "train")], ["0"]) == 1
    assert len(seen) == 1


@pytest.mark.parametrize("gpus", [[], ["0", "0"]])
def test_invalid_gpu_assignments_are_rejected(gpus):
    with pytest.raises(ValueError):
        run_local([], gpus)
