"""bycycle 1.2 fits for one channel and one frequency band."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ..jobs import checkpoint
from scipy.signal import butter, detrend, sosfiltfilt

from ..io.recording import Recording
from .segments import accepted_runs, covered_length
from .prepare import prepare_recording
from .settings import NAMED_BANDS, FitSettings

BANDS: list[tuple[str, tuple[float, float]]] = [
    (f"{name} {low:g}–{high:g} Hz", (low, high)) for name, low, high in NAMED_BANDS
]


@dataclass(frozen=True)
class BurstCutoff:
    """How one burst rule treated the cycles in a fit."""

    key: str
    label: str
    cutoff: float
    median: float | None
    n_passed: int
    n_cycles: int


@dataclass(frozen=True)
class BurstEpisode:
    """One contiguous run of burst cycles. Times are seconds in the recording."""

    start_s: float
    end_s: float
    n_cycles: int
    period_ms: float | None
    volt_amp: float | None


@dataclass
class CycleFit:
    channel: str
    channel_index: int
    band: tuple[float, float]
    sfreq: float
    filtered: np.ndarray
    n_cycles: int
    n_burst_cycles: int
    n_burst_episodes: int
    burst_duration_s: float
    burst_fraction: float | None
    period_ms: float | None
    rise_decay: float | None
    peak_trough: float | None
    volt_amp: float | None
    burst_period_ms: float | None
    burst_rise_decay: float | None
    burst_peak_trough: float | None
    burst_volt_amp: float | None
    spans: list[tuple[int, int, bool]] = field(default_factory=list)
    peaks: list[int] = field(default_factory=list)
    cycles: list[dict] = field(default_factory=list)
    cutoffs: list[BurstCutoff] = field(default_factory=list)
    episodes: list[BurstEpisode] = field(default_factory=list)
    t_offset: float = 0.0
    burst_note: str = ""
    notes: str = ""
    failures: list[dict] = field(default_factory=list)

    @property
    def n_bursts(self) -> int:
        """Cycles that belong to a burst. Prefer ``n_burst_cycles``; this is not an episode count."""
        return self.n_burst_cycles


def fit_cycles(
    recording: Recording,
    channel_index: int,
    band: tuple[float, float],
    settings: FitSettings | None = None,
) -> CycleFit:
    """Run :class:`bycycle.Bycycle` on one channel."""
    from bycycle import Bycycle

    if not 0 <= channel_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    recording.require_analysis_channel(channel_index)
    low, high = float(band[0]), float(band[1])
    nyquist = recording.sfreq / 2.0
    if low <= 0 or high <= low:
        raise ValueError("The cycle band must be a positive frequency range.")
    if high >= nyquist:
        raise ValueError(
            f"The {low:g}–{high:g} Hz band is above the Nyquist frequency ({nyquist:.1f} Hz)."
        )

    chosen = settings or FitSettings()
    chosen.validate()
    t_offset = min(max(0.0, float(chosen.t_start)), recording.duration_s)
    original_notes = list(recording.notes)
    recording = prepare_recording(recording, chosen)
    preparation = " ".join(
        note for note in recording.notes
        if note not in original_notes and not note.startswith("Fit window")
    )

    # WinEEG files can carry a large linear drift. Each accepted run is detrended
    # on its own, and a BAD or nonfinite gap is never part of a cycle.
    signal = np.asarray(recording.data[channel_index], dtype=np.float64)
    mask = recording.mask(channel_index) & np.isfinite(signal)
    filtered = np.full(signal.shape, np.nan, dtype=np.float64)
    model = Bycycle(
        center_extrema="peak",
        burst_method="cycles",
        thresholds=chosen.cycle_thresholds(),
        return_samples=True,
    )
    frames = []
    episode_rows: list[BurstEpisode] = []
    minimum = max(int(round(recording.sfreq)), int(round(3.0 * recording.sfreq / low)))
    failures = []
    runs = list(accepted_runs(mask))
    for position, (start, stop) in enumerate(runs):
        checkpoint(position, len(runs), f"Cycle run {position + 1} of {len(runs)}")
        if stop - start < minimum:
            continue
        piece = signal[start:stop]
        if not np.isfinite(piece).all():
            continue
        piece = detrend(piece, type="linear")
        try:
            filtered[start:stop] = _bandpass(piece, recording.sfreq, low, high)
            model.fit(piece, recording.sfreq, (low, high))
        except Exception as exc:
            failures.append({"start_s": t_offset + start / recording.sfreq, "end_s": t_offset + stop / recording.sfreq, "reason": str(exc)})
            filtered[start:stop] = np.nan
            continue
        piece_frame = model.df_features
        if piece_frame is None or not len(piece_frame):
            continue
        piece_frame = piece_frame.copy()
        for column in piece_frame.columns:
            if str(column).startswith("sample_"):
                piece_frame[column] = piece_frame[column] + start
        if "is_burst" in piece_frame:
            episode_rows.extend(burst_episodes(piece_frame, recording.sfreq, t_offset))
        frames.append(piece_frame)
    frame = _concat_frames(frames)
    spans: list[tuple[int, int, bool]] = []
    peaks: list[int] = []
    burst_mask = np.zeros(0, dtype=bool)
    if frame is not None and len(frame):
        burst_mask = np.asarray(frame["is_burst"], dtype=bool) if "is_burst" in frame else np.zeros(len(frame), dtype=bool)
        for start, stop, is_burst, peak in zip(
            frame["sample_last_trough"],
            frame["sample_next_trough"],
            burst_mask,
            frame["sample_peak"],
        ):
            spans.append((int(start), int(stop), bool(is_burst)))
            peaks.append(int(peak))
    cycle_rows = _cycle_rows(frame, recording.sfreq)

    def _median(column: str, mask: np.ndarray | None = None) -> float | None:
        if frame is None or not len(frame) or column not in frame:
            return None
        values = np.asarray(frame[column], dtype=np.float64)
        if mask is not None:
            values = values[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.median(values))

    period = _median("period")
    burst_period = _median("period", burst_mask)
    n_burst_cycles = int(burst_mask.sum()) if burst_mask.size else 0
    covered = covered_length(spans)
    accepted = int(mask.sum())
    fraction = None if accepted <= 0 else covered / accepted
    thresholds = chosen.cycle_thresholds()
    note = "" if n_burst_cycles else burst_note(frame, thresholds)
    return CycleFit(
        channel=recording.ch_names[channel_index],
        channel_index=channel_index,
        band=(low, high),
        sfreq=float(recording.sfreq),
        filtered=filtered,
        n_cycles=len(spans),
        n_burst_cycles=n_burst_cycles,
        n_burst_episodes=len(episode_rows),
        burst_duration_s=covered / recording.sfreq,
        burst_fraction=fraction,
        period_ms=None if period is None else period / recording.sfreq * 1000.0,
        rise_decay=_median("time_rdsym"),
        peak_trough=_median("time_ptsym"),
        volt_amp=_median("volt_amp"),
        burst_period_ms=None if burst_period is None else burst_period / recording.sfreq * 1000.0,
        burst_rise_decay=_median("time_rdsym", burst_mask),
        burst_peak_trough=_median("time_ptsym", burst_mask),
        burst_volt_amp=_median("volt_amp", burst_mask),
        spans=spans,
        peaks=peaks,
        cycles=cycle_rows,
        cutoffs=burst_cutoffs(frame, thresholds),
        episodes=episode_rows,
        t_offset=t_offset,
        burst_note=note,
        notes=preparation,
        failures=failures,
    )


_BURST_CUTOFFS = (
    ("amplitude fraction", "amp_fraction", "amp_fraction_threshold"),
    ("amplitude consistency", "amp_consistency", "amp_consistency_threshold"),
    ("period consistency", "period_consistency", "period_consistency_threshold"),
    ("monotonicity", "monotonicity", "monotonicity_threshold"),
)


def burst_cutoffs(frame, thresholds: dict) -> list[BurstCutoff]:
    """Median and pass count for each burst cutoff. A pass is strictly above the cutoff."""
    if frame is None or len(frame) == 0:
        return []
    rows = []
    count = len(frame)
    for label, column, key in _BURST_CUTOFFS:
        if column not in frame or key not in thresholds:
            continue
        values = np.asarray(frame[column], dtype=np.float64)
        finite = values[np.isfinite(values)]
        cutoff = float(thresholds[key])
        rows.append(BurstCutoff(
            key=column,
            label=label,
            cutoff=cutoff,
            median=None if finite.size == 0 else float(np.median(finite)),
            n_passed=int(np.sum(np.isfinite(values) & (values > cutoff))),
            n_cycles=count,
        ))
    return rows


def burst_episodes(frame, sfreq: float, t_offset: float = 0.0) -> list[BurstEpisode]:
    """Contiguous burst runs in one fitted piece. Adjacent pieces are not joined."""
    if frame is None or len(frame) == 0 or "is_burst" not in frame or sfreq <= 0:
        return []
    mask = np.asarray(frame["is_burst"], dtype=bool)
    starts = np.asarray(frame["sample_last_trough"], dtype=np.float64)
    stops = np.asarray(frame["sample_next_trough"], dtype=np.float64)
    periods = np.asarray(frame["period"], dtype=np.float64) if "period" in frame else None
    volts = np.asarray(frame["volt_amp"], dtype=np.float64) if "volt_amp" in frame else None
    found: list[BurstEpisode] = []
    index = 0
    count = len(mask)
    while index < count:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and mask[end + 1]:
            end += 1
        chosen = slice(index, end + 1)
        found.append(BurstEpisode(
            start_s=t_offset + float(starts[index]) / sfreq,
            end_s=t_offset + float(stops[end]) / sfreq,
            n_cycles=end - index + 1,
            period_ms=_finite_median(None if periods is None else periods[chosen], sfreq),
            volt_amp=_finite_median(None if volts is None else volts[chosen]),
        ))
        index = end + 1
    return found


def _finite_median(values: np.ndarray | None, sfreq: float | None = None) -> float | None:
    if values is None:
        return None
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    median = float(np.median(finite))
    if sfreq is not None:
        return median / sfreq * 1000.0
    return median


def burst_note(frame, thresholds: dict) -> str:
    """Why a fit with cycles still has no bursts. Empty when nothing blocked them."""
    if frame is None or len(frame) == 0:
        return ""
    minimum = int(thresholds.get("min_n_cycles", 3))
    count = len(frame)
    if count < 3:
        return (
            f"No burst: only {count} cycles. The first and last never count, "
            f"and a burst needs {minimum} in a row."
        )

    passed: list[np.ndarray] = []
    blocked: list[str] = []
    for label, column, key in _BURST_CUTOFFS:
        if column not in frame or key not in thresholds:
            continue
        values = np.asarray(frame[column], dtype=np.float64)
        cutoff = float(thresholds[key])
        ok = np.isfinite(values) & (values > cutoff)
        passed.append(ok)
        if not bool(np.any(ok[1:-1])):
            blocked.append(f"{label} above {_cutoff_text(cutoff)}")
    if not passed:
        return "No burst under the current rule."

    together = np.logical_and.reduce(passed)
    together = np.array(together, dtype=bool, copy=True)
    together[0] = False
    together[-1] = False
    n_pass = int(together.sum())
    longest = _longest_run(together)
    if n_pass == 0:
        if blocked:
            return f"No burst: no cycle passed {_join(blocked)}."
        return "No burst: no cycle passed every cutoff at once."
    if longest < minimum:
        cycle_word = "cycle" if n_pass == 1 else "cycles"
        return (
            f"No burst: {n_pass} {cycle_word} pass every cutoff, "
            f"but the longest run is {longest} and a burst needs {minimum}."
        )
    return "No burst under the current rule."


def _longest_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _cutoff_text(value: float) -> str:
    return f"{value:g}"


def _concat_frames(frames: list):
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def _cycle_rows(frame, sfreq: float) -> list[dict]:
    if frame is None or not len(frame):
        return []
    rows = []
    for index in range(len(frame)):
        period = float(frame["period"].iloc[index])
        row = {
            "period_ms": period / sfreq * 1000.0,
            "rise_decay": float(frame["time_rdsym"].iloc[index]),
            "peak_trough": float(frame["time_ptsym"].iloc[index]),
            "volt_amp": float(frame["volt_amp"].iloc[index]),
            "is_burst": bool(frame["is_burst"].iloc[index]) if "is_burst" in frame else False,
        }
        for column in ("amp_fraction", "amp_consistency", "period_consistency", "monotonicity"):
            if column in frame:
                row[column] = float(frame[column].iloc[index])
        for column in ("sample_last_trough", "sample_next_trough", "sample_peak"):
            if column in frame:
                row[column] = int(frame[column].iloc[index])
        rows.append(row)
    return rows


def _bandpass(signal: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = butter(3, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal)
