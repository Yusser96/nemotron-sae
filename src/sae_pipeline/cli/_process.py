"""Process-exit helper shared by CLI entry points."""

from __future__ import annotations

import os
import sys
from typing import Callable


def run_then_hard_exit(main: Callable[[], None]) -> None:
    """Run ``main``, then bypass interpreter shutdown.

    Triton can leave an autotuner worker thread alive after the CUDA work is
    done, and normal interpreter shutdown then aborts in PyGILState_Release.
    Call this only from ``if __name__ == "__main__":``: by the time ``main``
    returns normally, every checkpoint, cache shard and log has already been
    committed, so skipping shutdown is safe. An exception from ``main``
    propagates as usual; ``os._exit`` is never reached in that case.
    """
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
