"""Fit choices used for resting EEG with specparam and bycycle.

The defaults match the app's original fit: the whole file, 1–40 Hz, a 4 s
Welch segment (0.25 Hz), a fixed aperiodic slope, and the bycycle library
burst rule. The other presets are the ranges and thresholds used in resting
and eyes-open/eyes-closed papers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

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
    # Selected band for the reactivity peak. In the window this is also the bycycle band.
    alpha_low: float = 8.0
    alpha_high: float = 12.0
    amp_fraction: float = 0.0
    amp_consistency: float = 0.5
    period_consistency: float = 0.5
    monotonicity: float = 0.8
    min_n_cycles: int = 3
    # 0 keeps every sample. The default drops samples outside a normal EEG range.
    amplitude_limit_uv: float = 500.0
    # stored keeps the file reference. The others are explicit remontages.
    reference: str = "stored"
    # None fits the window as one spectrum. A value splits it into equal epochs.
    epoch_s: float | None = None

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (int, float)) and not isfinite(value):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be finite.")
        if self.t_start < 0:
            raise ValueError("The fit window cannot start before 0 s.")
        if self.t_length is not None and self.t_length <= 0:
            raise ValueError("The fit window length must be longer than 0 s.")
        if self.fmin <= 0 or self.fmax <= self.fmin:
            raise ValueError("The spectrum fit needs a low frequency below the high one.")
        if self.segment_s <= 0:
            raise ValueError("The Welch segment must be longer than 0 s.")
        if self.aperiodic_mode not in {"fixed", "knee"}:
            raise ValueError("The aperiodic mode is fixed or knee.")
        if self.peak_width_low <= 0 or self.peak_width_high <= self.peak_width_low:
            raise ValueError("Peak width needs a lower limit below the upper limit.")
        if self.max_n_peaks < 1 or int(self.max_n_peaks) != self.max_n_peaks:
            raise ValueError("At least one peak has to be allowed.")
        if self.alpha_low <= 0 or self.alpha_high <= self.alpha_low:
            raise ValueError("The band needs a low frequency below the high one.")
        if self.peak_threshold <= 0:
            raise ValueError("The peak threshold must be above 0.")
        if self.min_peak_height < 0:
            raise ValueError("The minimum peak height cannot be negative.")
        for name in ("amp_fraction", "amp_consistency", "period_consistency", "monotonicity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be between 0 and 1.")
        if self.min_n_cycles < 1 or int(self.min_n_cycles) != self.min_n_cycles:
            raise ValueError("A burst needs at least one cycle.")
        if self.amplitude_limit_uv < 0:
            raise ValueError("The amplitude limit cannot be negative. Use 0 to keep every sample.")
        if self.reference not in {
            "stored", "average", "linked", "mastoid", "left", "right", "vertex",
            "ipsilateral", "contralateral", "rest", "hjorth", "csd", "global",
        }:
            raise ValueError(
                "The reference is stored, average, linked ears, linked mastoids, left ear, right ear, "
                "vertex, ipsilateral ears, contralateral ears, REST, a local Laplacian, "
                "current source density, or global scale."
            )
        if self.epoch_s is not None and float(self.epoch_s) < 8.0:
            raise ValueError("An epoch must be at least 8 s.")

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
        """Selected band. The default is alpha, 8–12 Hz, and any other range is allowed."""
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
            round(self.amplitude_limit_uv, 3),
            self.reference,
            None if self.epoch_s is None else round(float(self.epoch_s), 3),
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

# Short name and the edges used by the fit bar, the cycle page, and reactivity.
NAMED_BANDS: list[tuple[str, float, float]] = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 12.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 45.0),
]


def band_label(low: float, high: float) -> str:
    """Short name for a range. 8–12 Hz is Alpha. Anything else unmatched is written as 7–13 Hz."""
    for name, edge_low, edge_high in NAMED_BANDS:
        if abs(float(low) - edge_low) < 0.05 and abs(float(high) - edge_high) < 0.05:
            return name
    return f"{float(low):g}–{float(high):g} Hz"


def reference_phrase(reference: str) -> str:
    """Short label for the reference used in a fit."""
    if reference == "average":
        return "average reference"
    if reference == "linked":
        return "linked-ear reference"
    if reference == "mastoid":
        return "linked-mastoid reference"
    if reference == "rest":
        return "REST, template spherical model"
    if reference == "hjorth":
        return "local Laplacian"
    if reference == "left":
        return "left-ear reference"
    if reference == "right":
        return "right-ear reference"
    if reference == "vertex":
        return "vertex reference"
    if reference == "ipsilateral":
        return "ipsilateral-ear reference"
    if reference == "contralateral":
        return "contralateral-ear reference"
    if reference == "csd":
        return "current source density"
    if reference == "global":
        return "average reference with the global scale factor"
    return "reference as stored"


def amplitude_unit(reference: str) -> str:
    """Unit of a sample after the remontage. Current source density is not microvolts."""
    if reference == "csd":
        return "µV/m²"
    if reference == "global":
        return "scaled µV"
    return "µV"


def power_unit(reference: str) -> str:
    """Unit of a power spectrum after the remontage.

    Current source density is a surface Laplacian, so power is µV²/(m⁴·Hz).
    Global scale divides that power by the factor and is labelled as scaled.
    """
    if reference == "csd":
        return "µV²/(m⁴·Hz)"
    if reference == "global":
        return "scaled µV²/Hz"
    return "µV²/Hz"


def format_number(value: float) -> str:
    """Print a measured number without rounding 8.4 or 31.5 up to the next whole value."""
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{rounded:.0f}"
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_hz(value: float) -> str:
    """Print a frequency without rounding 31.5 Hz up to 32."""
    return format_number(value)


def format_seconds(value: float) -> str:
    """Print a time in seconds without rounding 8.4 s up to 8 or 9."""
    return format_number(value)


def range_note(sfreq: float, fmax: float) -> str:
    """Explain a requested high frequency the sampling rate cannot represent."""
    limit = float(sfreq) / 2.0 - 0.5
    if float(fmax) <= limit + 1e-6:
        return ""
    return (
        f"Sampling rate {sfreq:g} Hz reaches {limit:g} Hz, "
        f"so the {fmax:g} Hz limit was brought down."
    )


def segment_note(sfreq: float, segment_s: float, nperseg: int) -> str:
    """Explain a Welch window shortened because the accepted stretch is shorter."""
    requested = int(round(float(sfreq) * float(segment_s)))
    if nperseg >= requested:
        return ""
    return (
        f"The Welch segment is {nperseg / float(sfreq):g} s ({nperseg} samples) "
        f"because no accepted stretch covers the {float(segment_s):g} s setting."
    )


def band_phrase(low: float, high: float) -> str:
    """Label with the range, such as Alpha 8–12 Hz or 7–13 Hz."""
    name = band_label(low, high)
    if name.isalpha():
        return f"{name} {float(low):g}–{float(high):g} Hz"
    return name


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
    return recording.cropped(i0, i1)


def matched_settings(
    settings: FitSettings,
    recording: Recording,
    others: list[Recording],
) -> FitSettings:
    """When the window is the whole file, fit every open recording to the shorter length.

    An explicit start or length is left as the user set it. A single open file is unchanged.
    """
    if settings.t_length is not None or settings.t_start > 0.0 or not others:
        return settings
    durations = [recording.duration_s, *[item.duration_s for item in others]]
    shorter = min(durations)
    if recording.duration_s <= shorter + 1e-9:
        return settings
    return replace(settings, t_length=shorter)


def window_bounds(recording: Recording, settings: FitSettings) -> tuple[float, float]:
    """Start and end, in seconds, of the samples a fit will use."""
    start = min(max(0.0, float(settings.t_start)), recording.duration_s)
    if settings.t_length is None:
        end = recording.duration_s
    else:
        end = min(recording.duration_s, start + float(settings.t_length))
    return start, end
