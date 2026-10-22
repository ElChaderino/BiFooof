"""Background workers so spectral fits do not block the trace view."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class CallWorker(QThread):
    succeeded = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, generation: int, fn, args: tuple = ()):
        super().__init__()
        self.generation = generation
        self._fn = fn
        self._args = args

    def run(self) -> None:
        try:
            self.succeeded.emit(self.generation, self._fn(*self._args))
        except Exception as exc:
            self.failed.emit(self.generation, "".join(traceback.format_exception(exc)))


def launch(owner, worker: CallWorker) -> CallWorker:
    """Start ``worker`` and keep it alive until the thread has finished.

    Replacing the only Python reference to a running ``QThread`` destroys it
    mid-fit. Each owner keeps a list until ``finished`` fires.
    """
    live = getattr(owner, "_live_workers", None)
    if live is None:
        live = []
        owner._live_workers = live
    live.append(worker)
    worker.finished.connect(lambda w=worker: _forget(owner, w))
    worker.start()
    return worker


def wait_for(owner, timeout_ms: int = 20000) -> None:
    """Block until this owner's background fits have stopped."""
    for worker in list(getattr(owner, "_live_workers", ())):
        if worker.isRunning():
            worker.wait(timeout_ms)


def _forget(owner, worker: CallWorker) -> None:
    live = getattr(owner, "_live_workers", None)
    if live and worker in live:
        live.remove(worker)
