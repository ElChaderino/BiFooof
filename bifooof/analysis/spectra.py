"""Welch power spectra in log10 units for specparam."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def welch_spectra(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs, linear_power)`` with power shaped ``(n_channels, n_freqs)``.

    Power is in microvolts squared per Hz. specparam logs it during the fit.
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

    target = int(round(sfreq / 0.25))
    nperseg = int(min(data.shape[1], max(target, 256)))
    nperseg = min(nperseg, data.shape[1])
    if nperseg < 16:
        raise ValueError("The recording is too short to estimate a spectrum.")

    freqs, power = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    if int(mask.sum()) < 8:
        raise ValueError("Not enough frequency bins in the requested range.")
    freqs = np.asarray(freqs[mask], dtype=np.float64)
    power = np.maximum(np.asarray(power[:, mask], dtype=np.float64), np.finfo(np.float64).tiny)
    return freqs, power
