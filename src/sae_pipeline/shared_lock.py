"""Cross-client locks using atomic directory creation, without advisory flock.

BeeGFS can configure flock/fcntl to be local to one client. Never steal a
directory lock on timeout: a paused owner could still be writing. Recovery
must establish that its Slurm allocation and processes have ended first.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time
import uuid


class DirectoryLock:
    def __init__(self, path):
        self.path = Path(str(path) + '.dlock')
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self, *, blocking=True, timeout=60.0):
        if self.acquired:
            raise RuntimeError('Shared locks are not reentrant')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.path.mkdir()
            except FileExistsError:
                if not blocking:
                    return False
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Shared lock busy: {self.path}; inspect owner.json, never automatically steal it')
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            self.acquired = True
            # A crash here deliberately leaves a lock requiring explicit recovery.
            with (self.path / 'owner.json').open('x') as output:
                json.dump(dict(token=self.token, pid=os.getpid(), host=socket.gethostname(),
                               job_id=os.environ.get('SLURM_JOB_ID'), at=time.time()), output)
                output.flush()
                os.fsync(output.fileno())
            return True

    def close(self):
        if not self.acquired:
            return
        owner = self.path / 'owner.json'
        if json.loads(owner.read_text())['token'] != self.token:
            raise RuntimeError('Shared lock owner changed; refusing to release another owner')
        owner.unlink()
        self.path.rmdir()
        self.acquired = False


@contextmanager
def file_lock(path, *, timeout=60.0):
    lock = DirectoryLock(path)
    lock.acquire(timeout=timeout)
    try:
        yield lock
    finally:
        lock.close()
