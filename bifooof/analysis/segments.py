"""Contiguous runs of accepted samples."""

from __future__ import annotations

import numpy as np


def accepted_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[start, stop)`` pairs where ``mask`` is true."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def episode_count(mask: np.ndarray) -> int:
    """How many contiguous true runs are in ``mask``."""
    return len(accepted_runs(mask))


def covered_length(spans: list[tuple[int, int, bool]]) -> int:
    """Sample count covered by burst spans, merging overlaps."""
    chosen = sorted((start, stop) for start, stop, is_burst in spans if is_burst and stop > start)
    if not chosen:
        return 0
    total = 0
    current_start, current_stop = chosen[0]
    for start, stop in chosen[1:]:
        if start <= current_stop:
            current_stop = max(current_stop, stop)
        else:
            total += current_stop - current_start
            current_start, current_stop = start, stop
    return total + current_stop - current_start
