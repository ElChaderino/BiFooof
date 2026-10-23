"""Phase–amplitude coupling as a separate research measure.

Nothing in the ordinary spectrum or burst fit imports this. A modulation index
is not a band power and not a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

from ..io.recording import Recording
from .cycles import _bandpass
from .prepare import prepare_recording
from .segments import accepted_runs
from .settings import FitSettings


@dataclass
class PacResult:
    mi: float | None
    surrogate_mean: float | None
    surrogate_sd: float | None
    p_value: float | None
    n_surrogates: int
    phase_band: tuple[float, float]
    amp_band: tuple[float, float]
    histogram: np.ndarray | None
    bin_edges: np.ndarray | None
    note: str = ""
    comodulogram: np.ndarray | None = None
    phase_centers: np.ndarray | None = None
    amp_centers: np.ndarray | None = None
    accepted_s: float = 0.0
    n_cells: int = 0
    warnings: list[str] = field(default_factory=list)


def modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> tuple[float, np.ndarray, np.ndarray]:
    """Tort's modulation index. Returns the index, the bin means, and the bin edges."""
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    which = np.digitize(phase, edges) - 1
    which = np.clip(which, 0, n_bins - 1)
    means = np.zeros(n_bins, dtype=np.float64)
    for bin_index in range(n_bins):
        chosen = amplitude[which == bin_index]
        if chosen.size:
            means[bin_index] = float(np.mean(chosen))
    total = float(means.sum())
    if total <= 0 or not np.isfinite(total):
        return float("nan"), means, edges
    probability = means / total
    kept = probability[probability > 0]
    entropy = float(-np.sum(kept * np.log(kept)))
    index = (np.log(n_bins) - entropy) / np.log(n_bins)
    return float(index), means, edges


def phase_amplitude(
    recording: Recording,
    settings: FitSettings,
    channel_index: int,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_surrogates: int = 200,
    minimum_s: float = 60.0,
    comodulogram_grid: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 1,
    amp_index: int | None = None,
    nonsinusoidal: bool = False,
) -> PacResult:
    """Modulation index, with time-shifted surrogates.

    Phase comes from ``channel_index``. Amplitude comes from ``amp_index``,
    or from the same channel when that is left empty.
    ``minimum_s`` is 60 s in the window. Tests pass a shorter floor.
    The comodulogram, when requested, is the raw index on a coarse grid.
    Its note names how many cells were tested. Surrogates are computed for the
    selected bands only, and the map is not a corrected test of every cell.
    A harmonic in the amplitude band skips the index only when the current
    cycles are also far from symmetric. A pure periodic envelope does not rise
    above a time shift, because the shift only moves which phase bin is tall.
    """
    if not 0 <= channel_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    amplitude_index = channel_index if amp_index is None else int(amp_index)
    if not 0 <= amplitude_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    _check_band(phase_band, "phase")
    _check_band(amp_band, "amplitude")
    chosen = settings
    chosen.validate()
    prepared = prepare_recording(recording, chosen)
    phase_signal = np.asarray(prepared.data[channel_index], dtype=np.float64)
    amp_signal = np.asarray(prepared.data[amplitude_index], dtype=np.float64)
    mask = (
        prepared.mask(channel_index) & prepared.mask(amplitude_index)
        & np.isfinite(phase_signal) & np.isfinite(amp_signal)
    )
    runs = [pair for pair in accepted_runs(mask) if pair[1] - pair[0] >= int(prepared.sfreq)]
    if not runs:
        return _empty(phase_band, amp_band, n_surrogates, "No accepted stretch to measure.")
    start, stop = max(runs, key=lambda pair: pair[1] - pair[0])
    accepted_s = (stop - start) / prepared.sfreq
    if accepted_s < float(minimum_s):
        return _empty(
            phase_band, amp_band, n_surrogates,
            f"Needs about {float(minimum_s):.0f} s of accepted signal. This stretch is {accepted_s:.0f} s.",
            accepted_s,
        )
    harmonic = _harmonic_reason(phase_band, amp_band)
    if harmonic and nonsinusoidal:
        return _empty(
            phase_band, amp_band, n_surrogates,
            harmonic + " The current cycles are far from symmetric, so the index was not computed.",
            accepted_s,
        )
    warnings = []
    if harmonic:
        warnings.append(
            harmonic + " The current cycles are not far from symmetric, so the index is still shown."
        )
    try:
        phase, amplitude = _phase_and_amplitude(
            phase_signal[start:stop], amp_signal[start:stop], prepared.sfreq, phase_band, amp_band,
        )
    except ValueError as exc:
        return _empty(phase_band, amp_band, n_surrogates, str(exc), accepted_s)
    index, means, edges = modulation_index(phase, amplitude)
    surrogates = _surrogates(phase, amplitude, int(n_surrogates), prepared.sfreq, phase_band[0], seed)
    finite = surrogates[np.isfinite(surrogates)]
    p_value = None
    mean = sd = None
    if finite.size:
        mean = float(np.mean(finite))
        sd = float(np.std(finite))
        p_value = float((np.sum(finite >= index) + 1) / (finite.size + 1))
    result = PacResult(
        mi=None if not np.isfinite(index) else float(index),
        surrogate_mean=mean,
        surrogate_sd=sd,
        p_value=p_value,
        n_surrogates=int(finite.size),
        phase_band=(float(phase_band[0]), float(phase_band[1])),
        amp_band=(float(amp_band[0]), float(amp_band[1])),
        histogram=means,
        bin_edges=edges,
        accepted_s=accepted_s,
        note=(
            "Surrogates are time shifts of the amplitude. "
            "A purely periodic envelope does not rise above them, because a shift only moves which phase bin is tall."
        ),
        warnings=warnings,
    )
    if comodulogram_grid is not None:
        phase_centers, amp_centers = comodulogram_grid
        result.phase_centers = np.asarray(phase_centers, dtype=np.float64)
        result.amp_centers = np.asarray(amp_centers, dtype=np.float64)
        result.comodulogram = _grid(phase_signal[start:stop], prepared.sfreq, phase_centers, amp_centers)
        result.n_cells = int(result.comodulogram.size)
        finite = int(np.isfinite(result.comodulogram).sum())
        if finite == result.n_cells:
            counted = f"{result.n_cells} map cells"
        else:
            counted = f"{finite} of {result.n_cells} map cells"
        result.note = (
            "Surrogates are time shifts of the amplitude. "
            f"{counted}. The map is not corrected across those cells."
        )
    return result


def _phase_and_amplitude(phase_signal, amp_signal, sfreq, phase_band, amp_band):
    nyquist = sfreq / 2.0
    if amp_band[1] >= nyquist or phase_band[1] >= nyquist:
        raise ValueError("That band reaches the Nyquist frequency.")
    phase_signal = _bandpass(phase_signal, sfreq, float(phase_band[0]), float(phase_band[1]))
    amp_signal = _bandpass(amp_signal, sfreq, float(amp_band[0]), float(amp_band[1]))
    trim = int(round(sfreq / max(float(phase_band[0]), 0.5)))
    if phase_signal.size <= 2 * trim + 8:
        trim = 0
    phase = np.angle(hilbert(phase_signal))
    amplitude = np.abs(hilbert(amp_signal))
    if trim:
        phase = phase[trim:-trim]
        amplitude = amplitude[trim:-trim]
    return phase, amplitude


def _surrogates(phase, amplitude, count, sfreq, phase_low, seed) -> np.ndarray:
    if count <= 0:
        return np.zeros(0)
    generator = np.random.default_rng(seed)
    minimum = max(1, int(round(sfreq / max(float(phase_low), 0.5))))
    if amplitude.size <= minimum * 2 + 2:
        return np.zeros(0)
    found = np.empty(count, dtype=np.float64)
    for index in range(count):
        shift = int(generator.integers(minimum, amplitude.size - minimum))
        found[index] = modulation_index(phase, np.roll(amplitude, shift))[0]
    return found


def _grid(signal, sfreq, phase_centers, amp_centers) -> np.ndarray:
    matrix = np.full((amp_centers.size, phase_centers.size), np.nan, dtype=np.float64)
    phase_signals = []
    for center in phase_centers:
        low, high = float(center) - 1.0, float(center) + 1.0
        if low <= 0 or high >= sfreq / 2.0:
            phase_signals.append(None)
            continue
        try:
            phase_signals.append(np.angle(hilbert(_bandpass(signal, sfreq, low, high))))
        except ValueError:
            phase_signals.append(None)
    amp_signals = []
    for center in amp_centers:
        width = max(2.0, float(center) * 0.15)
        low, high = float(center) - width, float(center) + width
        if low <= 0 or high >= sfreq / 2.0:
            amp_signals.append(None)
            continue
        try:
            amp_signals.append(np.abs(hilbert(_bandpass(signal, sfreq, low, high))))
        except ValueError:
            amp_signals.append(None)
    for row, amplitude in enumerate(amp_signals):
        if amplitude is None:
            continue
        for column, phase in enumerate(phase_signals):
            if phase is None:
                continue
            matrix[row, column] = modulation_index(phase, amplitude)[0]
    return matrix


def _harmonic_reason(phase_band, amp_band) -> str:
    center = 0.5 * (float(phase_band[0]) + float(phase_band[1]))
    harmonic = 2.0 * center
    if float(amp_band[0]) <= harmonic <= float(amp_band[1]):
        return (
            "The amplitude band contains a harmonic of the phase band. "
            "A non-sinusoidal wave can imitate coupling."
        )
    return ""


def _check_band(band: tuple[float, float], name: str) -> None:
    low, high = float(band[0]), float(band[1])
    if low <= 0 or high <= low:
        raise ValueError(f"The {name} band needs a low frequency below the high one.")


def _empty(phase_band, amp_band, n_surrogates, note: str, accepted_s: float = 0.0) -> PacResult:
    return PacResult(
        mi=None,
        surrogate_mean=None,
        surrogate_sd=None,
        p_value=None,
        n_surrogates=int(n_surrogates),
        phase_band=(float(phase_band[0]), float(phase_band[1])),
        amp_band=(float(amp_band[0]), float(amp_band[1])),
        histogram=None,
        bin_edges=None,
        note=note,
        accepted_s=accepted_s,
    )
