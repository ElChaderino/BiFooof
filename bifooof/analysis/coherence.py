"""Coherence between two channels of one recording.

Both channels go through the same preparation as the spectrum fit, and only
samples accepted on both channels are used. A high value is a relationship
between the two signals, not a connection between brain regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import csd, detrend, welch

from ..io.recording import Recording
from .prepare import prepare_recording
from .segments import accepted_runs, welch_segment_count
from .settings import FitSettings, format_seconds, range_note, segment_note, window_bounds


@dataclass
class CoherenceResult:
    channel_a: str
    channel_b: str
    freqs_hz: np.ndarray
    coherence: np.ndarray
    phase_deg: np.ndarray
    imaginary: np.ndarray
    wpli: np.ndarray | None
    n_segments: int
    accepted_s: float
    reference: str
    note: str = ""


def coherence_pair(
    recording: Recording,
    settings: FitSettings,
    index_a: int,
    index_b: int,
) -> CoherenceResult:
    """Magnitude-squared coherence, cross-spectral phase, imaginary coherence, and wPLI.

    The Welch segment is the one on the fit bar, so the frequency bins match
    the spectrum. wPLI needs several segments; otherwise it is left empty.
    """
    if index_a == index_b:
        raise ValueError("Pick two different channels.")
    for index in (index_a, index_b):
        if not 0 <= index < recording.n_channels:
            raise IndexError("Channel index is outside the recording.")
        recording.require_analysis_channel(index)
    chosen = settings
    chosen.validate()
    prepared = prepare_recording(recording, chosen)
    left = np.asarray(prepared.data[index_a], dtype=np.float64)
    right = np.asarray(prepared.data[index_b], dtype=np.float64)
    mask = prepared.mask(index_a) & prepared.mask(index_b) & np.isfinite(left) & np.isfinite(right)
    target = int(round(prepared.sfreq * float(chosen.segment_s)))
    if target < 16:
        raise ValueError(
            f"A {float(chosen.segment_s):g} s Welch segment is {target} samples at {prepared.sfreq:g} Hz. "
            "Coherence needs at least 16 samples in the segment."
        )
    longest = max((stop - start for start, stop in accepted_runs(mask)), default=0)
    nperseg = min(target, longest)
    if nperseg < 16:
        raise ValueError("No accepted stretch is long enough for coherence.")

    sxx = syy = sxy = None
    weight = 0.0
    imag_rows: list[np.ndarray] = []
    freqs = None
    skipped_s = 0.0
    used_samples = 0
    noverlap = nperseg // 2
    for start, stop in accepted_runs(mask):
        if stop - start < nperseg:
            skipped_s += (stop - start) / prepared.sfreq
            continue
        used_samples += stop - start
        piece_a = detrend(left[start:stop], type="linear")
        piece_b = detrend(right[start:stop], type="linear")
        freqs_a, pxx = welch(piece_a, fs=prepared.sfreq, nperseg=nperseg, noverlap=noverlap)
        _freqs_b, pyy = welch(piece_b, fs=prepared.sfreq, nperseg=nperseg, noverlap=noverlap)
        _freqs_xy, pxy = csd(piece_a, piece_b, fs=prepared.sfreq, nperseg=nperseg, noverlap=noverlap)
        run_weight = float(welch_segment_count(stop - start, nperseg, noverlap))
        if sxx is None:
            sxx = np.zeros(pxx.shape, dtype=np.float64)
            syy = np.zeros(pyy.shape, dtype=np.float64)
            sxy = np.zeros(pxy.shape, dtype=np.complex128)
            freqs = np.asarray(freqs_a, dtype=np.float64)
        sxx += np.asarray(pxx, dtype=np.float64) * run_weight
        syy += np.asarray(pyy, dtype=np.float64) * run_weight
        sxy += np.asarray(pxy, dtype=np.complex128) * run_weight
        weight += run_weight
        hop = max(1, nperseg - noverlap)
        for offset in range(0, stop - start - nperseg + 1, hop):
            seg_a = piece_a[offset:offset + nperseg]
            seg_b = piece_b[offset:offset + nperseg]
            _seg_f, seg_xy = csd(seg_a, seg_b, fs=prepared.sfreq, nperseg=nperseg, noverlap=0)
            imag_rows.append(np.imag(np.asarray(seg_xy, dtype=np.complex128)))
    if sxx is None or sxy is None or syy is None or freqs is None or weight <= 0:
        raise ValueError("No accepted stretch is long enough for coherence.")
    sxx /= weight
    syy /= weight
    sxy /= weight
    denom = sxx * syy
    coherence = np.full(freqs.shape, np.nan, dtype=np.float64)
    imaginary = np.full(freqs.shape, np.nan, dtype=np.float64)
    usable = denom > 0
    coherence[usable] = (np.abs(sxy[usable]) ** 2) / denom[usable]
    imaginary[usable] = np.imag(sxy[usable]) / np.sqrt(denom[usable])
    phase = np.degrees(np.angle(sxy))
    phase[~usable] = np.nan
    wpli = None
    if len(imag_rows) >= 5:
        stacked = np.vstack(imag_rows)
        scale = np.mean(np.abs(stacked), axis=0)
        wpli = np.full(scale.shape, np.nan, dtype=np.float64)
        usable_w = scale > 0
        wpli[usable_w] = np.abs(np.mean(stacked[:, usable_w], axis=0)) / scale[usable_w]
    low = float(chosen.fmin)
    high = min(float(chosen.fmax), prepared.sfreq / 2.0 - 0.5)
    if high <= low:
        raise ValueError(
            f"Sampling rate {prepared.sfreq:g} Hz cannot cover {low:g}–{float(chosen.fmax):g} Hz."
        )
    chosen_freq = (freqs >= low) & (freqs <= high)
    if int(chosen_freq.sum()) < 2:
        raise ValueError("Not enough frequency bins in the requested range.")
    notes = [
        range_note(prepared.sfreq, float(chosen.fmax)),
        segment_note(prepared.sfreq, chosen.segment_s, nperseg),
    ]
    if prepared.reference_applied != chosen.reference and prepared.notes:
        notes.append(prepared.notes[-1])
    accepted_s = used_samples / prepared.sfreq
    if accepted_s < 3.0 * chosen.segment_s:
        notes.append("Few clean segments, so this coherence estimate is thin.")
    if skipped_s > 0.05:
        notes.append(
            f"{format_seconds(skipped_s)} s in shorter stretches was left out of the coherence."
        )
    note = " ".join(part for part in notes if part)
    _t0, _t1 = window_bounds(recording, chosen)
    return CoherenceResult(
        channel_a=prepared.ch_names[index_a],
        channel_b=prepared.ch_names[index_b],
        freqs_hz=freqs[chosen_freq],
        coherence=np.clip(coherence[chosen_freq], 0.0, 1.0),
        phase_deg=phase[chosen_freq],
        imaginary=imaginary[chosen_freq],
        wpli=None if wpli is None else wpli[chosen_freq],
        n_segments=len(imag_rows),
        accepted_s=accepted_s,
        reference=prepared.reference_applied,
        note=note,
    )


def zero_lag_mask(result: CoherenceResult, coherence_floor: float = 0.5) -> np.ndarray:
    """True where coherence is high and the imaginary part is near zero.

    That pattern is what volume conduction and a shared reference produce.
    It is not proof of either one.
    """
    return (result.coherence >= coherence_floor) & np.isfinite(result.imaginary) & (np.abs(result.imaginary) < 0.05)
