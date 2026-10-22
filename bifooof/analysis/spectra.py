"""Welch power spectra in log10 units for specparam."""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend, welch

from .segments import accepted_runs


def welch_spectra(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    segment_s: float = 4.0,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(freqs, linear_power, nperseg)``.

    Power is in microvolts squared per Hz. specparam logs it during the fit.
    A straight drift line is removed inside each accepted run. Samples marked
    invalid are not part of any run, so a BAD segment cannot enter the spectrum
    and a drift line is not fit across it.
    """
    if data.ndim != 2:
        raise ValueError("Expected channels-by-samples data.")
    if sfreq <= 0:
        raise ValueError("Sampling rate must be positive.")
    nyquist = sfreq / 2.0
    high = min(float(fmax), nyquist - 0.5)
    low = float(fmin)
    if high <= low + 1.0:
        raise ValueError(
            f"Sampling rate {sfreq:.1f} Hz is too low for a {fmin:g}–{fmax:g} Hz spectrum."
        )

    values = np.asarray(data, dtype=np.float64)
    if valid is not None:
        accepted = np.asarray(valid, dtype=bool)
        if accepted.shape != values.shape:
            raise ValueError("The validity mask must have the same shape as the data.")
    else:
        accepted = np.ones(values.shape, dtype=bool)
    accepted = accepted & np.isfinite(values)

    # 4 s is 0.25 Hz. Longer segments smooth the spectrum; specparam fits those better.
    target = int(round(sfreq * float(segment_s)))
    longest = _longest_run(accepted)
    nperseg = int(min(max(target, 256), longest if longest else 0))
    if nperseg < 16:
        raise ValueError("No accepted stretch is long enough to estimate a spectrum.")

    power = None
    freqs = None
    for index in range(values.shape[0]):
        pieces = []
        weights = []
        for start, stop in accepted_runs(accepted[index]):
            segment = values[index, start:stop]
            if segment.size < nperseg or not np.isfinite(segment).all():
                continue
            if segment.size >= 2:
                segment = detrend(segment, type="linear")
            segment_freqs, segment_power = welch(segment, fs=sfreq, nperseg=nperseg)
            pieces.append(np.asarray(segment_power, dtype=np.float64))
            weights.append(segment.size)
            freqs = segment_freqs
        if power is None and pieces:
            power = np.full((values.shape[0], pieces[0].size), np.nan, dtype=np.float64)
        if pieces and power is not None:
            power[index] = np.average(pieces, axis=0, weights=weights)
    if freqs is None:
        raise ValueError("No accepted stretch is long enough to estimate a spectrum.")
    mask = (freqs >= low) & (freqs <= high)
    if int(mask.sum()) < 8:
        raise ValueError("Not enough frequency bins in the requested range.")
    freqs = np.asarray(freqs[mask], dtype=np.float64)
    power = np.asarray(power[:, mask], dtype=np.float64)
    finite = np.isfinite(power)
    power = np.where(finite, np.maximum(power, np.finfo(np.float64).tiny), np.nan)
    return freqs, power, nperseg


def _longest_run(valid: np.ndarray) -> int:
    longest = 0
    for row in valid:
        for start, stop in accepted_runs(row):
            longest = max(longest, stop - start)
    return longest
