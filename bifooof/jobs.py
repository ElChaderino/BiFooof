"""Cooperative background tasks and a bounded result cache."""
from collections import OrderedDict
from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from threading import Event
import sys

import numpy as np


class Cancelled(Exception):
    pass


class JobToken:
    def __init__(self):
        self.stop = Event()
        self.progress = (0, 0, "Starting…")

    def checkpoint(self, done=0, total=0, message="Working…"):
        if self.stop.is_set():
            raise Cancelled("Cancelled at the next safe stopping point.")
        self.progress = (int(done), int(total), str(message))


current_job = ContextVar("bifooof_job", default=None)


def checkpoint(done=0, total=0, message="Working…"):
    token = current_job.get()
    if token is not None:
        token.checkpoint(done, total, message)


def memory_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(memory_size(k, seen) + memory_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(memory_size(v, seen) for v in value)
    if is_dataclass(value):
        return sum(memory_size(getattr(value, f.name), seen) for f in fields(value))
    return sys.getsizeof(value)


class ResultCache(MutableMapping):
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self._items = OrderedDict()

    def __getitem__(self, key):
        value, _size = self._items[key]
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._items:
            del self[key]
        size = memory_size(value)
        if size > self.max_bytes:
            return
        while self._items and self.bytes_used + size > self.max_bytes:
            self.__delitem__(next(iter(self._items)))
        self._items[key] = (value, size)
        self.bytes_used += size

    def __delitem__(self, key):
        _value, size = self._items.pop(key)
        self.bytes_used -= size

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


def display_envelope(values, max_points=2500):
    """Original sample indices retaining extrema and both sides of every gap."""
    values = np.asarray(values)
    if values.size <= max_points:
        return np.arange(values.size)
    width = max(1, int(np.ceil(values.size / max(1, max_points // 2))))
    indexes = [0, values.size - 1]
    for start in range(0, values.size, width):
        piece = values[start:start + width]
        finite = np.flatnonzero(np.isfinite(piece))
        if finite.size:
            indexes.extend((start + finite[np.argmin(piece[finite])], start + finite[np.argmax(piece[finite])]))
    changes = np.flatnonzero(np.diff(np.isfinite(values)))
    indexes.extend(changes)
    indexes.extend(changes + 1)
    return np.unique(indexes)
