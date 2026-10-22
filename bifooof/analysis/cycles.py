"""bycycle 1.2 fits for one channel and one frequency band."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

from ..io.recording import Recording

# Documented bycycle defaults. Passing them avoids the "no thresholds" warning
# while keeping the library's own burst rule.
_CYCLE_THRESHOLDS = {
    "amp_fraction_threshold": 0.0,
    "amp_consistency_threshold": 0.5,
    "period_consistency_threshold": 0.5,
    "monotonicity_threshold": 0.8,
    "min_n_cycles": 3,
}

BANDS: list[tuple[str, tuple[float, float]]] = [
    ("Delta 1–4 Hz", (1.0, 4.0)),
    ("Theta 4–8 Hz", (4.0, 8.0)),
    ("Alpha 8–12 Hz", (8.0, 12.0)),
    ("Beta 13–30 Hz", (13.0, 30.0)),
    ("Gamma 30–45 Hz", (30.0, 45.0)),
]


@dataclass
class CycleFit:
    channel: str
    channel_index: int
    band: tuple[float, float]
    sfreq: float
    filtered: np.ndarray
    n_cycles: int
    n_bursts: int
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


def fit_cycles(recording: Recording, channel_index: int, band: tuple[float, float]) -> CycleFit:
    """Run :class:`bycycle.Bycycle` on one channel."""
    from bycycle import Bycycle

    if not 0 <= channel_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    low, high = float(band[0]), float(band[1])
    nyquist = recording.sfreq / 2.0
    if low <= 0 or high <= low:
        raise ValueError("The cycle band must be a positive frequency range.")
    if high >= nyquist:
        raise ValueError(
            f"The {low:g}–{high:g} Hz band is above the Nyquist frequency ({nyquist:.1f} Hz)."
        )

    # WinEEG files can carry a large linear drift. Welch removes it before the
    # spectrum fit; bycycle does not, and the drift swallows the alpha cycles.
    signal = detrend(np.asarray(recording.data[channel_index], dtype=np.float64), type="linear")
    filtered = _bandpass(signal, recording.sfreq, low, high)
    model = Bycycle(
        center_extrema="peak",
        burst_method="cycles",
        thresholds=_CYCLE_THRESHOLDS,
        return_samples=True,
    )
    model.fit(signal, recording.sfreq, (low, high))
    frame = model.df_features
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
    return CycleFit(
        channel=recording.ch_names[channel_index],
        channel_index=channel_index,
        band=(low, high),
        sfreq=float(recording.sfreq),
        filtered=filtered,
        n_cycles=len(spans),
        n_bursts=int(burst_mask.sum()) if burst_mask.size else 0,
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
    )


def _cycle_rows(frame, sfreq: float) -> list[dict]:
    if frame is None or not len(frame):
        return []
    rows = []
    for index in range(len(frame)):
        period = float(frame["period"].iloc[index])
        rows.append({
            "period_ms": period / sfreq * 1000.0,
            "rise_decay": float(frame["time_rdsym"].iloc[index]),
            "peak_trough": float(frame["time_ptsym"].iloc[index]),
            "volt_amp": float(frame["volt_amp"].iloc[index]),
            "is_burst": bool(frame["is_burst"].iloc[index]) if "is_burst" in frame else False,
        })
    return rows


def _bandpass(signal: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = butter(3, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal)
