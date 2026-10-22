"""Coordination must work even when advisory file locks have no global effect."""
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest
from sae_pipeline.shared_lock import DirectoryLock, file_lock


def _increment(path, count, start):
    # No flock/fcntl is used: these stand for distinct BeeGFS clients.
    path = Path(path)
    start.wait(20)
    for _ in range(count):
        with file_lock(path.with_suffix('.lock')):
            value = json.loads(path.read_text())
            time.sleep(0.001)
            value['count'] += 1
            path.write_text(json.dumps(value))


def _crash_with_lock(path):
    lock = DirectoryLock(path)
    assert lock.acquire()
    os._exit(0)


def test_processes_do_not_lose_updates_without_advisory_locks(tmp_path):
    path = tmp_path / 'counter.json'
    path.write_text('{"count": 0}')
    context = multiprocessing.get_context('spawn')
    start = context.Event()
    processes = [context.Process(target=_increment, args=(str(path), 30, start)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(40)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert json.loads(path.read_text())['count'] == 120


def test_lock_owner_nonblocking_and_exception_release(tmp_path):
    path = tmp_path / 'ownership.lock'
    first = DirectoryLock(path)
    assert first.acquire()
    owner = json.loads((first.path / 'owner.json').read_text())
    assert owner['pid'] == os.getpid() and owner['host'] and owner['token'] == first.token
    second = DirectoryLock(path)
    assert not second.acquire(blocking=False)
    with pytest.raises(TimeoutError, match='never automatically steal'):
        second.acquire(timeout=0.02)
    first.close()
    assert second.acquire(blocking=False)
    second.close()
    with pytest.raises(ValueError):
        with file_lock(path):
            raise ValueError('worker failed')
    assert not first.path.exists()


def test_crashed_owner_is_not_silently_evicted(tmp_path):
    path = tmp_path / 'producer.lock'
    process = multiprocessing.get_context('spawn').Process(target=_crash_with_lock, args=(str(path),))
    process.start()
    process.join(20)
    assert process.exitcode == 0
    contender = DirectoryLock(path)
    with pytest.raises(TimeoutError):
        contender.acquire(timeout=0.02)
    assert contender.path.exists()


def test_owner_cannot_release_someone_elses_lock(tmp_path):
    lock = DirectoryLock(tmp_path / 'pool.lock')
    lock.acquire()
    owner = lock.path / 'owner.json'
    value = json.loads(owner.read_text())
    value['token'] = 'different-owner'
    owner.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='owner changed'):
        lock.close()
    assert lock.path.exists()
