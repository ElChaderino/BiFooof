"""Fit choices used for resting EEG with specparam and bycycle.

The defaults match the app's original fit: the whole file, 1–40 Hz, a 4 s
Welch segment (0.25 Hz), a fixed aperiodic slope, and the bycycle library
burst rule. The other presets are the ranges and thresholds used in resting
and eyes-open/eyes-closed papers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..io.recording import Recording


@dataclass
class FitSettings:
    """One analysis window and the specparam / bycycle knobs for it."""

    t_start: float = 0.0
    t_length: float | None = None
    fmin: float = 1.0
    fmax: float = 40.0
    segment_s: float = 4.0
    aperiodic_mode: str = "fixed"
    peak_width_low: float = 0.5
    peak_width_high: float = 12.0
    max_n_peaks: int = 6
    min_peak_height: float = 0.05
    peak_threshold: float = 2.0
    alpha_low: float = 8.0
    alpha_high: float = 12.0
    amp_fraction: float = 0.0
    amp_consistency: float = 0.5
    period_consistency: float = 0.5
    monotonicity: float = 0.8
    min_n_cycles: int = 3

    def validate(self) -> None:
        if self.t_start < 0:
            raise ValueError("The fit window cannot start before 0 s.")
        if self.t_length is not None and self.t_length <= 0:
            raise ValueError("The fit window length must be longer than 0 s.")
        if self.fmax <= self.fmin:
            raise ValueError("The spectrum fit needs a low frequency below the high one.")
        if self.segment_s <= 0:
            raise ValueError("The Welch segment must be longer than 0 s.")
        if self.aperiodic_mode not in {"fixed", "knee"}:
            raise ValueError("The aperiodic mode is fixed or knee.")
        if self.peak_width_high <= self.peak_width_low:
            raise ValueError("Peak width needs a lower limit below the upper limit.")
        if self.max_n_peaks < 1:
            raise ValueError("At least one peak has to be allowed.")
        if self.alpha_high <= self.alpha_low:
            raise ValueError("The alpha band needs a low frequency below the high one.")
        if self.peak_threshold <= 0:
            raise ValueError("The peak threshold must be above 0.")
        if self.min_n_cycles < 1:
            raise ValueError("A burst needs at least one cycle.")

    def model_kwargs(self) -> dict:
        return {
            "aperiodic_mode": self.aperiodic_mode,
            "periodic_mode": "gaussian",
            "verbose": False,
            "peak_width_limits": (float(self.peak_width_low), float(self.peak_width_high)),
            "max_n_peaks": int(self.max_n_peaks),
            "min_peak_height": float(self.min_peak_height),
            "peak_threshold": float(self.peak_threshold),
        }

    def alpha_band(self) -> tuple[float, float]:
        return (float(self.alpha_low), float(self.alpha_high))

    def cycle_thresholds(self) -> dict:
        return {
            "amp_fraction_threshold": float(self.amp_fraction),
            "amp_consistency_threshold": float(self.amp_consistency),
            "period_consistency_threshold": float(self.period_consistency),
            "monotonicity_threshold": float(self.monotonicity),
            "min_n_cycles": int(self.min_n_cycles),
        }

    def key(self) -> tuple:
        return (
            round(self.t_start, 3),
            None if self.t_length is None else round(self.t_length, 3),
            round(self.fmin, 3),
            round(self.fmax, 3),
            round(self.segment_s, 3),
            self.aperiodic_mode,
            round(self.peak_width_low, 3),
            round(self.peak_width_high, 3),
            int(self.max_n_peaks),
            round(self.min_peak_height, 4),
            round(self.peak_threshold, 3),
            round(self.alpha_low, 3),
            round(self.alpha_high, 3),
            round(self.amp_fraction, 3),
            round(self.amp_consistency, 3),
            round(self.period_consistency, 3),
            round(self.monotonicity, 3),
            int(self.min_n_cycles),
        )


# name, fmin, fmax, segment_s, aperiodic, width low, width high, max peaks, min height
RANGE_PRESETS: list[tuple] = [
    ("1–40 Hz resting", 1.0, 40.0, 4.0, "fixed", 0.5, 12.0, 6, 0.05),
    ("2–40 Hz published", 2.0, 40.0, 4.0, "fixed", 1.0, 8.0, 8, 0.10),
    ("3–30 Hz bands", 3.0, 30.0, 4.0, "fixed", 1.0, 8.0, 6, 0.05),
    ("2–20 Hz narrow", 2.0, 20.0, 4.0, "fixed", 2.0, 8.0, 6, 0.05),
    ("Custom", None, None, None, None, None, None, None, None),
]

TIME_PRESETS: list[tuple[str, float, float | None]] = [
    ("Whole recording", 0.0, None),
    ("First 30 s", 0.0, 30.0),
    ("First 60 s", 0.0, 60.0),
    ("First 2 min", 0.0, 120.0),
    ("Custom", 0.0, None),
]

SEGMENT_PRESETS: list[tuple[str, float]] = [
    ("2 s  ·  0.5 Hz", 2.0),
    ("4 s  ·  0.25 Hz", 4.0),
    ("8 s  ·  0.125 Hz", 8.0),
]

# amp fraction, amp consistency, period consistency, monotonicity, min cycles
BURST_PRESETS: list[tuple] = [
    ("Library", 0.0, 0.5, 0.5, 0.8, 3),
    ("Resting alpha", 0.2, 0.5, 0.5, 0.9, 2),
    ("Paper alpha", 0.2, 0.4, 0.4, 0.8, 3),
]


def clip_recording(recording: Recording, settings: FitSettings) -> Recording:
    """Return the samples inside the fit window. The full recording is kept when it already matches."""
    start = max(0.0, float(settings.t_start))
    i0 = int(round(start * recording.sfreq))
    if settings.t_length is None:
        i1 = recording.n_samples
    else:
        i1 = i0 + max(1, int(round(float(settings.t_length) * recording.sfreq)))
    i0 = min(max(i0, 0), recording.n_samples)
    i1 = min(max(i1, i0), recording.n_samples)
    if i1 <= i0:
        raise ValueError("That time window is past the end of the recording.")
    if i0 == 0 and i1 == recording.n_samples:
        return recording
    t0 = i0 / recording.sfreq
    t1 = i1 / recording.sfreq
    return Recording(
        data=recording.data[:, i0:i1],
        ch_names=list(recording.ch_names),
        sfreq=recording.sfreq,
        source_path=recording.source_path,
        source_format=recording.source_format,
        notes=[*recording.notes, f"Fit window {t0:.1f}–{t1:.1f} s."],
    )


def window_bounds(recording: Recording, settings: FitSettings) -> tuple[float, float]:
    """Start and end, in seconds, of the samples a fit will use."""
    start = min(max(0.0, float(settings.t_start)), recording.duration_s)
    if settings.t_length is None:
        end = recording.duration_s
    else:
        end = min(recording.duration_s, start + float(settings.t_length))
    return start, end
