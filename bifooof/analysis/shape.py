"""Cycle shape, burst lengths, and the amplitude–frequency cloud.

All three read a CycleFit that bycycle already produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cycles import CycleFit


@dataclass
class BurstStats:
    n_episodes: int
    durations_s: np.ndarray
    gaps_s: np.ndarray
    median_s: float | None
    q1_s: float | None
    q3_s: float | None
    per_minute: float | None
    occupancy: float | None
    median_cycles: float | None
    accepted_s: float | None


def burst_stats(fit: CycleFit) -> BurstStats:
    """Durations and gaps from the episode list. Overlaps are already merged in occupancy."""
    durations = np.array([episode.end_s - episode.start_s for episode in fit.episodes], dtype=np.float64)
    gaps = []
    for earlier, later in zip(fit.episodes, fit.episodes[1:]):
        gap = later.start_s - earlier.end_s
        if gap >= 0:
            gaps.append(gap)
    gap_values = np.asarray(gaps, dtype=np.float64)
    occupancy = None if fit.burst_fraction is None else float(fit.burst_fraction)
    accepted = None
    per_minute = None
    if occupancy is not None and occupancy > 0 and fit.burst_duration_s > 0:
        accepted = fit.burst_duration_s / occupancy
        per_minute = len(fit.episodes) / (accepted / 60.0) if accepted else None
    elif occupancy == 0:
        per_minute = 0.0
    cycles = np.array([episode.n_cycles for episode in fit.episodes], dtype=np.float64)
    return BurstStats(
        n_episodes=len(fit.episodes),
        durations_s=durations,
        gaps_s=gap_values,
        median_s=_quartile(durations, 50),
        q1_s=_quartile(durations, 25),
        q3_s=_quartile(durations, 75),
        per_minute=per_minute,
        occupancy=occupancy,
        median_cycles=None if cycles.size == 0 else float(np.median(cycles)),
        accepted_s=accepted,
    )


def cycle_cloud(fit: CycleFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instantaneous frequency, amplitude, and whether the cycle was in a burst."""
    freqs = []
    amps = []
    bursts = []
    for row in fit.cycles:
        period = row.get("period_ms")
        amp = row.get("volt_amp")
        if period is None or amp is None or not np.isfinite(period) or period <= 0 or not np.isfinite(amp):
            continue
        freqs.append(1000.0 / float(period))
        amps.append(float(amp))
        bursts.append(bool(row.get("is_burst")))
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(amps, dtype=np.float64),
        np.asarray(bursts, dtype=bool),
    )


def aligned_cycles(
    fit: CycleFit,
    align: str = "trough",
    bursts_only: bool = False,
    normalized: bool = False,
    limit: int = 400,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the time axis, a cycle-by-time matrix, and how many cycles were not drawn.

    Native time keeps each cycle's own length and aligns on the trough or the peak.
    The normalized view resamples trough-to-trough onto one axis from 0 to 1.
    """
    snippets: list[tuple[np.ndarray, int]] = []
    for (start, stop, is_burst), peak in zip(fit.spans, fit.peaks):
        if bursts_only and not is_burst:
            continue
        if stop - start < 4 or not 0 <= start < stop <= fit.filtered.size:
            continue
        wave = np.asarray(fit.filtered[start:stop], dtype=np.float64)
        if wave.size < 4 or not np.isfinite(wave).all():
            continue
        anchor = 0 if align != "peak" else int(peak) - int(start)
        if not 0 <= anchor < wave.size:
            continue
        snippets.append((wave, anchor))
    total = len(snippets)
    hidden = 0
    if total > limit:
        pick = np.linspace(0, total - 1, limit).astype(int)
        snippets = [snippets[int(index)] for index in pick]
        hidden = total - len(snippets)
    if not snippets:
        return np.zeros(0), np.zeros((0, 0)), 0
    if normalized:
        grid = np.linspace(0.0, 1.0, 80)
        rows = []
        for wave, _anchor in snippets:
            source = np.linspace(0.0, 1.0, wave.size)
            rows.append(np.interp(grid, source, wave))
        return grid, np.vstack(rows), hidden
    left = min(-anchor / fit.sfreq for _wave, anchor in snippets)
    right = max((wave.size - anchor) / fit.sfreq for wave, anchor in snippets)
    count = int(np.median([wave.size for wave, _anchor in snippets]))
    count = max(8, min(count, 400))
    grid = np.linspace(left, right, count)
    rows = []
    for wave, anchor in snippets:
        source = (np.arange(wave.size) - anchor) / fit.sfreq
        rows.append(np.interp(grid, source, wave, left=np.nan, right=np.nan))
    return grid, np.vstack(rows), hidden


def _quartile(values: np.ndarray, percent: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percent))
