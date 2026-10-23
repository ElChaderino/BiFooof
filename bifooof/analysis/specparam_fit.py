"""specparam 2.0 group fits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..io.recording import Recording
from .prepare import peak_at_width_floor, prepare_recording
from .settings import FitSettings, window_bounds
from .spectra import welch_spectra


@dataclass
class ChannelFit:
    name: str
    index: int
    usable: bool
    reason: str = ""
    offset: float | None = None
    knee: float | None = None
    exponent: float | None = None
    r_squared: float | None = None
    error: float | None = None
    peaks: list[tuple[float, float, float]] = field(default_factory=list)
    freqs: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    model: np.ndarray | None = None
    aperiodic: np.ndarray | None = None
    n_epochs: int = 0
    n_peak_epochs: int = 0
    # How many epochs joined each peak in ``peaks``. Empty when the fit is one spectrum.
    peak_counts: list[int] = field(default_factory=list)
    peak_at_width_floor: bool = False

    @property
    def dominant_peak(self) -> tuple[float, float, float] | None:
        if not self.peaks:
            return None
        return max(self.peaks, key=lambda peak: peak[1])


@dataclass
class SpectrumFit:
    channels: list[ChannelFit]
    fmin: float
    fmax: float
    t_start: float = 0.0
    t_end: float = 0.0
    segment_s: float = 4.0
    aperiodic_mode: str = "fixed"
    welch_nperseg: int = 0
    detrend: str = "linear"
    epoch_s: float | None = None
    epoch_count: int = 0
    notes: str = ""


def fit_spectra(
    recording: Recording,
    settings: FitSettings | None = None,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> SpectrumFit:
    """Fit a :class:`specparam.SpectralGroupModel` to every usable channel."""
    from specparam import SpectralGroupModel, SpectralModel

    chosen = settings or FitSettings(fmin=fmin, fmax=fmax)
    chosen.validate()
    if chosen.epoch_s:
        return _fit_epochs(recording, chosen)
    t_start, t_end = window_bounds(recording, chosen)
    original_notes = list(recording.notes)
    recording = prepare_recording(recording, chosen)
    preparation = _preparation_notes(original_notes, recording.notes)
    fmin = float(chosen.fmin)
    fmax = float(chosen.fmax)
    data = np.asarray(recording.data, dtype=np.float64)
    bad = set(recording.bad_channels)
    results = []
    candidates: list[int] = []
    for index, name in enumerate(recording.ch_names):
        if name in bad:
            results.append(ChannelFit(name=name, index=index, usable=False, reason="Bad channel."))
            continue
        mask = recording.mask(index)
        samples = data[index, mask]
        samples = samples[np.isfinite(samples)]
        if samples.size < 16 or float(np.std(samples)) <= 1e-6:
            if samples.size < 16 and name in recording.dropped_by_amplitude:
                reason = "Amplitude above the limit."
            else:
                reason = "No accepted samples." if samples.size < 16 else "Flat channel."
            results.append(ChannelFit(name=name, index=index, usable=False, reason=reason))
            continue
        results.append(ChannelFit(name=name, index=index, usable=True))
        candidates.append(index)
    if not candidates:
        raise ValueError("Every channel is flat or rejected, so there is no spectrum to fit.")

    masks = np.vstack([recording.mask(index) for index in candidates])
    freqs, power, nperseg = welch_spectra(
        data[candidates], recording.sfreq, fmin=fmin, fmax=fmax,
        segment_s=chosen.segment_s, valid=masks,
    )
    usable = []
    kept_rows = []
    for slot, channel_index in enumerate(candidates):
        if np.isfinite(power[slot]).all():
            usable.append(channel_index)
            kept_rows.append(slot)
            continue
        item = results[channel_index]
        item.usable = False
        item.reason = "Accepted samples are shorter than the Welch window."
    if not usable:
        raise ValueError("No accepted stretch is long enough for the Welch window.")
    power = power[kept_rows]
    kwargs = chosen.model_kwargs()
    group = SpectralGroupModel(**kwargs)
    try:
        group.fit(freqs, power, n_jobs=1, progress=None)
        _fill_from_group(group, results, usable)
    except Exception:
        _fill_one_by_one(SpectralModel, results, usable, freqs, power, kwargs)

    fitted = [item for item in results if item.usable and item.freqs is not None]
    if not fitted:
        reasons = [item.reason for item in results if item.reason]
        detail = reasons[-1] if reasons else "no spectrum could be modeled"
        raise ValueError(f"specparam could not fit any channel. {detail}")
    _mark_fit_quality(results, chosen)
    return SpectrumFit(
        channels=results,
        fmin=fmin,
        fmax=min(fmax, float(freqs[-1])),
        t_start=t_start,
        t_end=t_end,
        segment_s=float(chosen.segment_s),
        aperiodic_mode=chosen.aperiodic_mode,
        welch_nperseg=int(nperseg),
        notes=preparation,
    )


def log_residual(item: ChannelFit) -> tuple[np.ndarray, np.ndarray] | None:
    """Observed minus model, both already in log10 power. Empty when either curve is missing."""
    if item.freqs is None or item.spectrum is None or item.model is None:
        return None
    freqs = np.asarray(item.freqs, dtype=np.float64)
    residual = np.asarray(item.spectrum, dtype=np.float64) - np.asarray(item.model, dtype=np.float64)
    if residual.shape != freqs.shape:
        return None
    return freqs, residual


def _fill_from_group(group, results: list[ChannelFit], usable: list[int]) -> None:
    aperiodic = np.atleast_2d(np.asarray(group.get_params("aperiodic"), dtype=np.float64))
    peaks = _peak_table(group.get_params("periodic"))
    errors = np.atleast_1d(np.asarray(group.get_metrics("error"), dtype=np.float64))
    gof = np.atleast_1d(np.asarray(group.get_metrics("gof"), dtype=np.float64))
    for slot, channel_index in enumerate(usable):
        model = group.get_model(int(slot), regenerate=True)
        item = results[channel_index]
        _store_curves(item, model)
        if aperiodic.ndim == 2 and slot < aperiodic.shape[0]:
            item.offset, item.exponent, item.knee = _read_aperiodic(aperiodic[slot])
        item.error = float(errors[slot]) if slot < errors.size else None
        item.r_squared = float(gof[slot]) if slot < gof.size else None
        item.peaks = _peaks_for(peaks, slot)


def _fill_one_by_one(spectral_model, results, usable, freqs, power, kwargs: dict) -> None:
    for slot, channel_index in enumerate(usable):
        item = results[channel_index]
        model = spectral_model(**kwargs)
        try:
            model.fit(freqs, power[slot])
            _store_curves(item, model)
            aperiodic = np.atleast_1d(np.asarray(model.get_params("aperiodic"), dtype=np.float64))
            item.offset, item.exponent, item.knee = _read_aperiodic(aperiodic)
            metrics = model.get_metrics("all")
            flat = np.atleast_1d(np.asarray(metrics, dtype=np.float64)).ravel()
            if flat.size >= 2:
                item.error = float(flat[0])
                item.r_squared = float(flat[1])
            item.peaks = [(cf, pw, bw) for cf, pw, bw in _peak_rows(model.get_params("periodic"))]
        except Exception as exc:
            item.usable = False
            item.reason = str(exc)


def _store_curves(item: ChannelFit, model) -> None:
    components = model.results.model
    item.freqs = np.asarray(model.data.freqs, dtype=np.float64).copy()
    item.spectrum = np.asarray(model.data.power_spectrum, dtype=np.float64).copy()
    item.model = np.asarray(components.get_component("full", "log"), dtype=np.float64).copy()
    item.aperiodic = np.asarray(components.get_component("aperiodic", "log"), dtype=np.float64).copy()


def _read_aperiodic(row) -> tuple[float | None, float | None, float | None]:
    """Return offset, exponent, and knee. Knee mode is offset, knee, exponent."""
    values = np.atleast_1d(np.asarray(row, dtype=np.float64)).ravel()
    finite = values[np.isfinite(values)]
    if finite.size >= 3:
        return float(values[0]), float(values[2]), float(values[1])
    if finite.size >= 2:
        return float(values[0]), float(values[1]), None
    return None, None, None


def _peak_table(values) -> np.ndarray:
    peaks = np.asarray([] if values is None else values, dtype=np.float64)
    if peaks.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    peaks = np.atleast_2d(peaks)
    if peaks.shape[1] == 3:
        index = np.zeros((peaks.shape[0], 1), dtype=np.float64)
        peaks = np.hstack([peaks, index])
    return peaks


def _peak_rows(values) -> list[tuple[float, float, float]]:
    table = _peak_table(values)
    rows = []
    for row in table:
        if row.size < 3 or not np.all(np.isfinite(row[:3])):
            continue
        rows.append((float(row[0]), float(row[1]), float(row[2])))
    return rows


def _peaks_for(table: np.ndarray, slot: int) -> list[tuple[float, float, float]]:
    if table.size == 0:
        return []
    chosen = table[np.rint(table[:, -1]).astype(int) == int(slot)]
    return [(float(row[0]), float(row[1]), float(row[2])) for row in chosen if np.all(np.isfinite(row[:3]))]


def _preparation_notes(before: list[str], after: list[str]) -> str:
    added = [note for note in after if note not in before and not note.startswith("Fit window")]
    return " ".join(added)


def _mark_fit_quality(results: list[ChannelFit], settings: FitSettings) -> None:
    low, high = settings.alpha_band()
    for item in results:
        item.n_epochs = 1 if item.usable else 0
        in_band = [peak for peak in item.peaks if low <= peak[0] <= high]
        item.n_peak_epochs = 1 if in_band else 0
        chosen = max(in_band, key=lambda peak: peak[1]) if in_band else item.dominant_peak
        item.peak_at_width_floor = bool(
            chosen is not None and peak_at_width_floor(chosen[2], settings.peak_width_low)
        )


def _fit_epochs(recording: Recording, settings: FitSettings) -> SpectrumFit:
    """Fit equal epochs and keep a peak only from the epochs where it is present."""
    t_start, t_end = window_bounds(recording, settings)
    prepared = prepare_recording(recording, settings)
    child = replace(
        settings,
        epoch_s=None,
        t_start=0.0,
        t_length=None,
        reference="stored",
        amplitude_limit_uv=0.0,
    )
    width = max(1, int(round(float(settings.epoch_s) * prepared.sfreq)))
    if prepared.n_samples < width:
        single = fit_spectra(prepared, child)
        single.t_start, single.t_end = t_start, t_end
        single.epoch_s = None
        single.epoch_count = 0
        extra = (
            f"The window is shorter than one {float(settings.epoch_s):.0f} s epoch, "
            "so it was fit as one spectrum."
        )
        preparation = _preparation_notes(list(recording.notes), prepared.notes)
        single.notes = " ".join(part for part in (preparation, extra) if part)
        return single
    pieces = []
    start = 0
    while start + width <= prepared.n_samples:
        pieces.append((start, start + width))
        start += width
    fits = []
    for i0, i1 in pieces:
        try:
            fits.append(fit_spectra(prepared.cropped(i0, i1), child))
        except ValueError:
            continue
    if not fits:
        raise ValueError("No epoch was long enough to fit.")
    combined = _combine_epochs(fits, settings, t_start, t_end)
    notes = _preparation_notes(list(recording.notes), prepared.notes)
    leftover = prepared.n_samples - start
    if leftover > 0 and prepared.sfreq:
        seconds = leftover / prepared.sfreq
        shown = f"{seconds:.0f}" if seconds >= 10 else f"{seconds:.1f}"
        notes = " ".join(
            part for part in (
                notes,
                f"The last {shown} s was shorter than one {float(settings.epoch_s):.0f} s epoch and was left out.",
            ) if part
        )
    combined.notes = notes
    return combined


def _combine_epochs(fits: list[SpectrumFit], settings: FitSettings, t_start: float, t_end: float) -> SpectrumFit:
    names = [item.name for item in fits[0].channels]
    channels = []
    for name in names:
        found = [next(item for item in fit.channels if item.name == name) for fit in fits]
        channels.append(_combine_channel(found, settings))
    if not any(item.usable for item in channels):
        raise ValueError("specparam could not fit any channel in these epochs.")
    return SpectrumFit(
        channels=channels,
        fmin=fits[0].fmin,
        fmax=fits[0].fmax,
        t_start=t_start,
        t_end=t_end,
        segment_s=float(settings.segment_s),
        aperiodic_mode=settings.aperiodic_mode,
        welch_nperseg=fits[0].welch_nperseg,
        epoch_s=float(settings.epoch_s or 0.0),
        epoch_count=len(fits),
    )


def _combine_channel(items: list[ChannelFit], settings: FitSettings) -> ChannelFit:
    index = items[0].index if items else 0
    name = items[0].name if items else ""
    usable = [
        item for item in items
        if item.usable and item.freqs is not None and item.spectrum is not None
    ]
    if not usable:
        reason = next((item.reason for item in items if item.reason), "No accepted samples.")
        return ChannelFit(name=name, index=index, usable=False, reason=reason)
    base = usable[0]
    same = [
        item for item in usable
        if item.freqs is not None and item.freqs.shape == base.freqs.shape and np.allclose(item.freqs, base.freqs)
    ]
    curves = same or usable
    low, high = settings.alpha_band()
    grouped = _group_peaks([item.peaks for item in usable])
    peaks = [(center, power, width) for center, power, width, _count in grouped]
    in_band = [peak for peak in grouped if low <= peak[0] <= high]
    chosen = max(in_band, key=lambda peak: peak[1]) if in_band else None
    # The count belongs to the reported in-band peak, not to every in-band center in the epoch.
    n_peak = 0 if chosen is None else int(chosen[3])
    floor_source = chosen if chosen is not None else (grouped[0] if grouped else None)
    aperiodics = [item.aperiodic for item in curves if item.aperiodic is not None and item.aperiodic.shape == base.spectrum.shape]
    models = [item.model for item in curves if item.model is not None and item.model.shape == base.spectrum.shape]
    return ChannelFit(
        name=name,
        index=index,
        usable=True,
        offset=_median_value(item.offset for item in usable),
        knee=_median_value(item.knee for item in usable),
        exponent=_median_value(item.exponent for item in usable),
        r_squared=_median_value(item.r_squared for item in usable),
        error=_median_value(item.error for item in usable),
        peaks=peaks,
        peak_counts=[count for _center, _power, _width, count in grouped],
        freqs=np.asarray(base.freqs, dtype=np.float64).copy(),
        spectrum=np.mean([item.spectrum for item in curves], axis=0),
        model=None if not models else np.mean(models, axis=0),
        aperiodic=None if not aperiodics else np.mean(aperiodics, axis=0),
        n_epochs=len(usable),
        n_peak_epochs=n_peak,
        peak_at_width_floor=bool(
            floor_source is not None and peak_at_width_floor(floor_source[2], settings.peak_width_low)
        ),
    )


EPOCH_PEAKS = (
    "Epochs do not overlap, and each one is fit on its own. "
    "Peaks whose centers fall within 1 Hz are one group. "
    "The reported center, height, and width are the medians of the epochs in that group. "
    "An epoch that missed the peak is left out of the median. It is not entered as a height of zero. "
    "The number of epochs with an in-band peak is counted separately and does not scale the height. "
    "Offset, exponent, and R² are medians across epochs that fit. The spectrum curve is their mean."
)


def epochs_for_peak(item: ChannelFit, peak: tuple[float, float, float] | None) -> int | None:
    """How many epochs contained this peak. None when the fit was one spectrum."""
    if peak is None or item.n_epochs <= 1:
        return None
    if item.peak_counts and len(item.peak_counts) == len(item.peaks):
        for stored, count in zip(item.peaks, item.peak_counts):
            if all(abs(float(stored[index]) - float(peak[index])) <= 1e-9 for index in range(3)):
                return int(count)
        return 0
    return item.n_peak_epochs


def _median_peaks(epoch_peaks: list[list[tuple[float, float, float]]], tol: float = 1.0) -> list[tuple[float, float, float]]:
    """Group peaks within 1 Hz and keep the median of each group."""
    return [(center, power, width) for center, power, width, _count in _group_peaks(epoch_peaks, tol)]


def _group_peaks(
    epoch_peaks: list[list[tuple[float, float, float]]],
    tol: float = 1.0,
) -> list[tuple[float, float, float, int]]:
    """Return ``(center, height, width, n_epochs)`` for each peak group.

    A group takes at most one peak from each epoch, the one closest to the
    group's current center. An epoch with no peak in the group does not
    contribute a zero height. The count is how many epochs joined that group.
    """
    clusters: list[dict] = []
    for epoch_index, peaks in enumerate(epoch_peaks):
        for cf, power, bandwidth in sorted(peaks, key=lambda peak: peak[1], reverse=True):
            best = None
            best_distance = None
            for cluster in clusters:
                if epoch_index in cluster["epochs"]:
                    continue
                distance = abs(float(cf) - float(np.median(cluster["centers"])))
                if distance <= tol and (best_distance is None or distance < best_distance):
                    best = cluster
                    best_distance = distance
            if best is None:
                clusters.append({
                    "centers": [float(cf)],
                    "powers": [float(power)],
                    "widths": [float(bandwidth)],
                    "epochs": {epoch_index},
                })
                continue
            best["centers"].append(float(cf))
            best["powers"].append(float(power))
            best["widths"].append(float(bandwidth))
            best["epochs"].add(epoch_index)
    combined = [
        (
            float(np.median(cluster["centers"])),
            float(np.median(cluster["powers"])),
            float(np.median(cluster["widths"])),
            len(cluster["epochs"]),
        )
        for cluster in clusters
    ]
    combined.sort(key=lambda peak: peak[1], reverse=True)
    return combined


def _median_value(values) -> float | None:
    chosen = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not chosen:
        return None
    return float(np.median(np.asarray(chosen, dtype=np.float64)))
