"""Spectrogram and sliding-window specparam.

Both run on the recording ``prepare_recording`` already clipped, re-referenced,
and cleaned. Times are seconds from the start of the open recording, the same
clock as a burst episode. The Welch segment on the fit bar is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import detrend, spectrogram

from ..io.recording import Recording
from .compare import alpha_peak
from .prepare import R2_WEAK, prepare_recording
from .segments import accepted_runs
from .settings import FitSettings, window_bounds
from .specparam_fit import ChannelFit, SpectrumFit, fit_spectra
from .spectra import welch_spectra


@dataclass
class Spectrogram:
    channel: str
    channel_index: int
    times_s: np.ndarray
    freqs_hz: np.ndarray
    power_db: np.ndarray
    window_s: float
    overlap: float
    nperseg: int
    noverlap: int
    df_hz: float
    dt_s: float
    t_start: float
    t_end: float
    notes: str = ""


@dataclass
class SlidingFit:
    windows: list[SpectrumFit]
    window_s: float
    overlap: float
    artifact_fraction: list[float]
    notes: str = ""


@dataclass
class ChannelSeries:
    """One channel across the sliding windows. Missing peaks are NaN."""

    channel: str
    times_s: np.ndarray
    exponent: np.ndarray
    offset: np.ndarray
    peak_hz: np.ndarray
    peak_height: np.ndarray
    peak_bandwidth: np.ndarray
    r_squared: np.ndarray
    weak: np.ndarray
    at_floor: np.ndarray
    artifact_fraction: np.ndarray
    knee: np.ndarray | None = None


@dataclass
class PeakTrack:
    times: list[float] = field(default_factory=list)
    centers: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    bandwidths: list[float] = field(default_factory=list)


def channel_spectrogram(
    recording: Recording,
    settings: FitSettings,
    channel_index: int,
    window_s: float = 2.0,
    overlap: float = 0.75,
) -> Spectrogram:
    """Hann spectrogram of one channel, in dB relative to 1 µV²/Hz.

    Each accepted run is detrended on its own. A rejected stretch stays blank
    instead of being bridged by a taper.
    """
    if not 0 <= channel_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    if window_s <= 0:
        raise ValueError("The spectrogram window has to be longer than 0 s.")
    if not 0 <= float(overlap) < 0.95:
        raise ValueError("Spectrogram overlap has to be below 95%.")
    chosen = settings
    chosen.validate()
    t_start, t_end = window_bounds(recording, chosen)
    prepared = prepare_recording(recording, chosen)
    signal = np.asarray(prepared.data[channel_index], dtype=np.float64)
    mask = prepared.mask(channel_index) & np.isfinite(signal)
    nperseg = max(16, int(round(prepared.sfreq * float(window_s))))
    noverlap = int(round(nperseg * float(overlap)))
    noverlap = min(max(noverlap, 0), nperseg - 1)
    hop = nperseg - noverlap
    if int(mask.sum()) < nperseg:
        raise ValueError("No accepted stretch is long enough for that spectrogram window.")

    columns: list[tuple[float, np.ndarray]] = []
    freqs = None
    for start, stop in accepted_runs(mask):
        if stop - start < nperseg:
            continue
        piece = detrend(signal[start:stop], type="linear")
        piece_freqs, _times, power = spectrogram(
            piece,
            fs=prepared.sfreq,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        if freqs is None:
            freqs = np.asarray(piece_freqs, dtype=np.float64)
        origin = t_start + start / prepared.sfreq
        for index, moment in enumerate(np.asarray(_times, dtype=np.float64)):
            columns.append((origin + float(moment), np.asarray(power[:, index], dtype=np.float64)))
    if freqs is None or not columns:
        raise ValueError("No accepted stretch is long enough for that spectrogram window.")

    columns.sort(key=lambda item: item[0])
    times = np.array([item[0] for item in columns], dtype=np.float64)
    grid = np.column_stack([item[1] for item in columns])
    if times.size > 1:
        gaps = np.diff(times)
        typical = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else hop / prepared.sfreq
        kept_t = []
        kept_p = []
        for index, moment in enumerate(times):
            if index and moment - times[index - 1] > 1.5 * typical:
                kept_t.append(0.5 * (times[index - 1] + moment))
                kept_p.append(np.full(grid.shape[0], np.nan))
            kept_t.append(float(moment))
            kept_p.append(grid[:, index])
        times = np.asarray(kept_t, dtype=np.float64)
        grid = np.column_stack(kept_p)

    high = min(float(chosen.fmax), prepared.sfreq / 2.0 - 0.5)
    low = float(chosen.fmin)
    chosen_freq = (freqs >= low) & (freqs <= high)
    if int(chosen_freq.sum()) < 2:
        raise ValueError("Not enough frequency bins in the requested range.")
    freqs = freqs[chosen_freq]
    grid = grid[chosen_freq]
    finite = np.isfinite(grid) & (grid > 0)
    decibels = np.full(grid.shape, np.nan, dtype=np.float64)
    decibels[finite] = 10.0 * np.log10(grid[finite])
    df = prepared.sfreq / nperseg
    dt = hop / prepared.sfreq
    return Spectrogram(
        channel=prepared.ch_names[channel_index],
        channel_index=channel_index,
        times_s=times,
        freqs_hz=freqs,
        power_db=decibels,
        window_s=float(window_s),
        overlap=float(overlap),
        nperseg=nperseg,
        noverlap=noverlap,
        df_hz=float(df),
        dt_s=float(dt),
        t_start=t_start,
        t_end=t_end,
        notes=(
            f"Frequency bins {df:.2f} Hz. Time step {dt:.2f} s. "
            "A burst shorter than the window is spread across that window."
        ),
    )


def sliding_spectra(
    recording: Recording,
    settings: FitSettings,
    window_s: float = 30.0,
    overlap: float = 0.5,
    channel_index: int | None = None,
) -> SlidingFit:
    """Fit specparam on successive windows and keep every window.

    This is not the fit bar's epoch mode. Epochs are averaged into one spectrum.
    Here each window stays a separate fit, and a window that cannot be fit stays
    in the list so the time axis does not close up.
    """
    if window_s <= 0:
        raise ValueError("The sliding window has to be longer than 0 s.")
    if not 0 <= float(overlap) < 0.95:
        raise ValueError("Sliding-window overlap has to be below 95%.")
    chosen = settings
    chosen.validate()
    t_start, _t_end = window_bounds(recording, chosen)
    prepared = prepare_recording(recording, chosen)
    width = max(1, int(round(float(window_s) * prepared.sfreq)))
    hop = max(1, int(round(width * (1.0 - float(overlap)))))
    if prepared.n_samples < width:
        pieces = [(0, prepared.n_samples)]
        note = f"Shorter than {float(window_s):.0f} s, so one window was fit."
    else:
        pieces = []
        start = 0
        while start + width <= prepared.n_samples:
            pieces.append((start, start + width))
            start += hop
        note = ""
        if chosen.segment_s * 2 > window_s:
            note = (
                f"A {float(window_s):.0f} s window with a {chosen.segment_s:g} s Welch segment "
                "has few averages."
            )
    child = replace(
        chosen,
        epoch_s=None,
        t_start=0.0,
        t_length=None,
        reference="stored",
        amplitude_limit_uv=0.0,
    )
    windows: list[SpectrumFit] = []
    fractions: list[float] = []
    mask = None if prepared.valid is None else prepared.valid
    for i0, i1 in pieces:
        fractions.append(_fraction(mask, prepared.data, i0, i1, channel_index))
        piece = prepared.cropped(i0, i1)
        if channel_index is not None:
            piece = _only(piece, channel_index)
        start_s = t_start + i0 / prepared.sfreq
        end_s = t_start + i1 / prepared.sfreq
        try:
            fit = fit_spectra(piece, child)
        except ValueError as exc:
            fit = _unfitted(piece, child, str(exc))
        fit.t_start = start_s
        fit.t_end = end_s
        windows.append(fit)
    if not windows:
        raise ValueError("No window was long enough to fit.")
    return SlidingFit(
        windows=windows,
        window_s=float(window_s),
        overlap=float(overlap),
        artifact_fraction=fractions,
        notes=note,
    )


def channel_series(sliding: SlidingFit, channel: str, band: tuple[float, float]) -> ChannelSeries:
    """Exponent, offset, and the in-band peak. A missing peak is NaN."""
    times = []
    exponent = []
    offset = []
    knee = []
    center = []
    height = []
    bandwidth = []
    scored = []
    weak = []
    floor = []
    for fit, fraction in zip(sliding.windows, sliding.artifact_fraction):
        times.append(0.5 * (fit.t_start + fit.t_end))
        item = _named(fit, channel)
        peak = None if item is None else alpha_peak(item, band)
        exponent.append(_finite(None if item is None else item.exponent))
        offset.append(_finite(None if item is None else item.offset))
        knee.append(_finite(None if item is None else item.knee))
        scored.append(_finite(None if item is None else item.r_squared))
        if peak is None:
            center.append(np.nan)
            height.append(np.nan)
            bandwidth.append(np.nan)
            floor.append(False)
        else:
            center.append(float(peak[0]))
            height.append(float(peak[1]))
            bandwidth.append(float(peak[2]))
            floor.append(bool(item is not None and item.peak_at_width_floor))
        r_value = None if item is None else item.r_squared
        weak.append(bool(r_value is not None and r_value < R2_WEAK))
    return ChannelSeries(
        channel=channel,
        times_s=np.asarray(times, dtype=np.float64),
        exponent=np.asarray(exponent, dtype=np.float64),
        offset=np.asarray(offset, dtype=np.float64),
        peak_hz=np.asarray(center, dtype=np.float64),
        peak_height=np.asarray(height, dtype=np.float64),
        peak_bandwidth=np.asarray(bandwidth, dtype=np.float64),
        r_squared=np.asarray(scored, dtype=np.float64),
        weak=np.asarray(weak, dtype=bool),
        at_floor=np.asarray(floor, dtype=bool),
        artifact_fraction=np.asarray(sliding.artifact_fraction, dtype=np.float64),
        knee=np.asarray(knee, dtype=np.float64),
    )


def track_peaks(sliding: SlidingFit, channel: str, band: tuple[float, float]) -> list[PeakTrack]:
    """Follow in-band peaks across windows.

    A peak continues when its center stays within 1.5 Hz, or half its previous
    bandwidth, of the last point. A window with no in-band peak ends the track.
    A taller peak outside that jump starts its own track.
    """
    closed: list[PeakTrack] = []
    active: list[PeakTrack] = []
    low, high = float(band[0]), float(band[1])
    for fit in sliding.windows:
        item = _named(fit, channel)
        mid = 0.5 * (fit.t_start + fit.t_end)
        peaks = []
        if item is not None and item.usable:
            peaks = [peak for peak in item.peaks if low <= peak[0] <= high]
        if not peaks:
            closed.extend(active)
            active = []
            continue
        claimed = [False] * len(peaks)
        still: list[PeakTrack] = []
        for track in active:
            last_cf = track.centers[-1]
            last_bw = track.bandwidths[-1]
            jump = max(1.5, float(last_bw) / 2.0)
            best = None
            best_dist = None
            for index, peak in enumerate(peaks):
                if claimed[index]:
                    continue
                dist = abs(peak[0] - last_cf)
                if dist <= jump and (best_dist is None or dist < best_dist):
                    best = index
                    best_dist = dist
            if best is None:
                closed.append(track)
                continue
            claimed[best] = True
            cf, pw, bw = peaks[best]
            track.times.append(mid)
            track.centers.append(float(cf))
            track.heights.append(float(pw))
            track.bandwidths.append(float(bw))
            still.append(track)
        for index, peak in enumerate(peaks):
            if claimed[index]:
                continue
            cf, pw, bw = peak
            still.append(PeakTrack(
                times=[mid],
                centers=[float(cf)],
                heights=[float(pw)],
                bandwidths=[float(bw)],
            ))
        active = still
    closed.extend(active)
    return closed


def track_summary(tracks: list[PeakTrack], n_windows: int, n_with_peak: int) -> str:
    """Center range, median height, and median width of the tallest track.

    The percent is how many windows had any in-band peak, not how many fell on one track.
    """
    if n_windows <= 0:
        return ""
    share = 100.0 * n_with_peak / n_windows
    present = f"An in-band peak is present in {share:.0f}% of windows."
    usable = [track for track in tracks if track.centers]
    if not usable:
        return present
    chosen = max(usable, key=lambda track: float(np.median(track.heights)))
    centers = np.asarray(chosen.centers, dtype=np.float64)
    heights = np.asarray(chosen.heights, dtype=np.float64)
    widths = np.asarray(chosen.bandwidths, dtype=np.float64)
    return (
        f"Tallest track {float(np.min(centers)):.1f}–{float(np.max(centers)):.1f} Hz, "
        f"median height {float(np.median(heights)):.2f}, median width {float(np.median(widths)):.2f} Hz. "
        f"{present}"
    )


def window_at(sliding: SlidingFit, center_s: float) -> SpectrumFit | None:
    """The sliding fit that contains ``center_s``. Overlaps keep the nearer center.

    None means the click is outside every window, so the Spectrum tab is still
    the fit-bar window.
    """
    containing = [
        fit for fit in sliding.windows
        if fit.t_start <= float(center_s) <= fit.t_end
    ]
    if not containing:
        return None
    return min(containing, key=lambda fit: abs(0.5 * (fit.t_start + fit.t_end) - float(center_s)))


def exponent_spread(sliding: SlidingFit) -> tuple[list[str], list[float | None], list[float | None], list[int]]:
    """Median exponent, median absolute deviation, and how many windows were usable.

    A channel with fewer than three usable windows is left empty so the head
    map does not color a site from one noisy fit.
    """
    if not sliding.windows:
        return [], [], [], []
    names = [item.name for item in sliding.windows[0].channels]
    medians: list[float | None] = []
    spreads: list[float | None] = []
    counts: list[int] = []
    for name in names:
        values = []
        for fit in sliding.windows:
            item = _named(fit, name)
            if item is None or not item.usable or item.exponent is None or not np.isfinite(item.exponent):
                continue
            values.append(float(item.exponent))
        counts.append(len(values))
        if len(values) < 3:
            medians.append(None)
            spreads.append(None)
            continue
        chosen = np.asarray(values, dtype=np.float64)
        middle = float(np.median(chosen))
        medians.append(middle)
        spreads.append(float(np.median(np.abs(chosen - middle))))
    return names, medians, spreads, counts


def block_band_power(
    recording: Recording,
    settings: FitSettings,
    channel_index: int,
    band: tuple[float, float],
    block_s: float = 2.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch power in ``band`` for successive blocks. Times are block centers."""
    if not 0 <= channel_index < recording.n_channels:
        raise IndexError("Channel index is outside the recording.")
    if block_s <= 0:
        raise ValueError("The block has to be longer than 0 s.")
    chosen = settings
    chosen.validate()
    t_start, _t_end = window_bounds(recording, chosen)
    prepared = prepare_recording(recording, chosen)
    signal = np.asarray(prepared.data[channel_index], dtype=np.float64)
    mask = prepared.mask(channel_index) & np.isfinite(signal)
    width = max(8, int(round(float(block_s) * prepared.sfreq)))
    hop = max(1, int(round(width * (1.0 - float(overlap)))))
    segment = min(float(chosen.segment_s), float(block_s))
    times = []
    power = []
    low, high = float(band[0]), float(band[1])
    for start in range(0, prepared.n_samples - width + 1, hop):
        stop = start + width
        if int(mask[start:stop].sum()) < max(16, width // 2):
            times.append(t_start + (start + width / 2) / prepared.sfreq)
            power.append(np.nan)
            continue
        piece = signal[start:stop]
        valid = mask[start:stop]
        try:
            freqs, spectra, _nperseg = welch_spectra(
                piece[None, :], prepared.sfreq, fmin=min(low, chosen.fmin), fmax=max(high, chosen.fmax),
                segment_s=segment, valid=valid[None, :],
            )
        except ValueError:
            times.append(t_start + (start + width / 2) / prepared.sfreq)
            power.append(np.nan)
            continue
        chosen_bins = (freqs >= low) & (freqs <= high) & np.isfinite(spectra[0])
        times.append(t_start + (start + width / 2) / prepared.sfreq)
        if not np.any(chosen_bins):
            power.append(np.nan)
        else:
            power.append(float(np.mean(spectra[0, chosen_bins])))
    return np.asarray(times, dtype=np.float64), np.asarray(power, dtype=np.float64)


def align_to_events(
    times_s: np.ndarray,
    values: np.ndarray,
    events_s: list[float],
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample each event onto one grid of seconds relative to the event."""
    if times_s.size == 0 or not events_s:
        return np.zeros(0), np.zeros((0, 0))
    ordered = np.sort(np.asarray(times_s, dtype=np.float64))
    diffs = np.diff(ordered)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if diffs.size == 0:
        step = 1.0
    else:
        smallest = float(np.min(diffs))
        close = diffs[diffs <= smallest * 3.0]
        step = float(np.median(close if close.size else diffs))
    step = max(step, 0.05)
    grid = np.arange(-float(pre_s), float(post_s) + step * 0.5, step)
    curves = []
    order = np.argsort(times_s)
    moments = np.asarray(times_s, dtype=np.float64)[order]
    series = np.asarray(values, dtype=np.float64)[order]
    for event in events_s:
        sample = np.interp(grid + float(event), moments, series, left=np.nan, right=np.nan)
        # interp fills across gaps. Blank samples that had no nearby real point.
        if moments.size:
            for index, moment in enumerate(grid + float(event)):
                if np.min(np.abs(moments - moment)) > step * 1.6:
                    sample[index] = np.nan
        curves.append(sample)
    return grid, np.vstack(curves)


def _fraction(mask: np.ndarray | None, data: np.ndarray, i0: int, i1: int, channel: int | None) -> float:
    if mask is None:
        piece = np.isfinite(data[:, i0:i1] if channel is None else data[channel, i0:i1])
    elif channel is None:
        piece = mask[:, i0:i1]
    else:
        piece = mask[channel, i0:i1]
    if piece.size == 0:
        return 1.0
    return float(1.0 - np.mean(piece))


def _only(recording: Recording, index: int) -> Recording:
    name = recording.ch_names[index]
    valid = None if recording.valid is None else recording.valid[index:index + 1]
    return Recording(
        data=recording.data[index:index + 1],
        ch_names=[name],
        sfreq=recording.sfreq,
        source_path=recording.source_path,
        source_format=recording.source_format,
        notes=list(recording.notes),
        marks=list(recording.marks),
        valid=valid,
        bad_channels=[item for item in recording.bad_channels if item == name],
        rejected_spans=list(recording.rejected_spans),
        reference_channels=dict(recording.reference_channels),
        dropped_by_amplitude=[item for item in recording.dropped_by_amplitude if item == name],
    )


def _unfitted(recording: Recording, settings: FitSettings, reason: str) -> SpectrumFit:
    return SpectrumFit(
        channels=[
            ChannelFit(name=name, index=index, usable=False, reason=reason)
            for index, name in enumerate(recording.ch_names)
        ],
        fmin=float(settings.fmin),
        fmax=float(settings.fmax),
        segment_s=float(settings.segment_s),
        aperiodic_mode=settings.aperiodic_mode,
        notes=reason,
    )


def _named(fit: SpectrumFit, channel: str) -> ChannelFit | None:
    for item in fit.channels:
        if item.name == channel:
            return item
    return None


def _finite(value: float | None) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    return float(value)
