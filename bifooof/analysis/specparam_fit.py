"""specparam 2.0 group fits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.recording import Recording
from .settings import FitSettings, clip_recording, window_bounds
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
    t_start, t_end = window_bounds(recording, chosen)
    recording = clip_recording(recording, chosen)
    fmin = float(chosen.fmin)
    fmax = float(chosen.fmax)
    data = np.asarray(recording.data, dtype=np.float64)
    usable = [index for index in range(data.shape[0]) if float(np.std(data[index])) > 1e-6]
    results = [
        ChannelFit(name=name, index=index, usable=index in usable,
                   reason="" if index in usable else "Flat channel.")
        for index, name in enumerate(recording.ch_names)
    ]
    if not usable:
        raise ValueError("Every channel is flat, so there is no spectrum to fit.")

    freqs, power = welch_spectra(
        data[usable], recording.sfreq, fmin=fmin, fmax=fmax, segment_s=chosen.segment_s,
    )
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
    return SpectrumFit(
        channels=results,
        fmin=fmin,
        fmax=min(fmax, float(freqs[-1])),
        t_start=t_start,
        t_end=t_end,
        segment_s=float(chosen.segment_s),
        aperiodic_mode=chosen.aperiodic_mode,
    )


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
            model.fit(freqs, power[slot], n_jobs=1, progress=None)
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
