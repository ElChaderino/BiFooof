"""Background workers so spectral fits do not block the trace view."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal
from ..jobs import Cancelled, JobToken, current_job


class CallWorker(QThread):
    succeeded = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, generation: int, fn, args: tuple = ()):
        super().__init__()
        self.generation = generation
        self._fn = fn
        self._args = args
        self.token = JobToken()
        self.scope = None
        self.pending = False

    def cancel(self):
        self.token.stop.set()
        if self.pending:
            self.pending = False
            self.failed.emit(self.generation, "Cancelled before starting.")

    def run(self) -> None:
        context = current_job.set(self.token)
        try:
            self.token.checkpoint()
            result = self._fn(*self._args)
            self.token.checkpoint(message="Complete")
            self.succeeded.emit(self.generation, result)
        except Cancelled as exc:
            self.failed.emit(self.generation, str(exc))
        except Exception as exc:
            self.failed.emit(self.generation, "".join(traceback.format_exception(exc)))
        finally:
            current_job.reset(context)


def launch(owner, worker: CallWorker, key=None) -> CallWorker:
    """Start ``worker`` and keep it alive until the thread has finished.

    Replacing the only Python reference to a running ``QThread`` destroys it
    mid-fit. Each owner keeps a list until ``finished`` fires.
    """
    live = getattr(owner, "_live_workers", None)
    if live is None:
        live = []
        owner._live_workers = live
    pending = getattr(owner, "_pending_by_scope", None)
    if pending is None:
        pending = owner._pending_by_scope = {}
    previous_pending = pending.pop(key, None)
    if previous_pending is not None:
        previous_pending.cancel()
        _forget(owner, previous_pending)
    running = [previous for previous in live if previous.scope == key and previous.isRunning()]
    for previous in running:
        if previous.isRunning():
            previous.cancel()
    worker.scope = key
    live.append(worker)
    if running:
        worker.pending = True
        pending[key] = worker
    else:
        _start(owner, worker)
    return worker


def _start(owner, worker):
    worker.pending = False
    worker.finished.connect(lambda w=worker: _finished(owner, w))
    worker.start()


def _finished(owner, worker):
    _forget(owner, worker)
    pending = getattr(owner, "_pending_by_scope", {}).pop(worker.scope, None)
    if pending is not None:
        if pending.token.stop.is_set():
            _forget(owner, pending)
        else:
            _start(owner, pending)


def wait_for(owner, timeout_ms: int = 20000) -> None:
    """Block until this owner's background fits have stopped."""
    for worker in list(getattr(owner, "_live_workers", ())):
        if worker.pending:
            worker.cancel()
        if worker.isRunning():
            worker.wait(timeout_ms)


def _forget(owner, worker: CallWorker) -> None:
    live = getattr(owner, "_live_workers", None)
    if live and worker in live:
        live.remove(worker)
