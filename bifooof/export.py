"""Save plots and fit tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import __version__
from .analysis.compare import ChannelComparison
from .analysis.cycles import CycleFit
from .analysis.settings import FitSettings, band_label
from .analysis.specparam_fit import SpectrumFit
from .io.condition import CONDITION_LABELS
from .io.recording import Recording


def save_figure(fig, path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    return destination


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".csv":
        destination = destination.with_suffix(".csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})
    return destination


def spectrum_rows(fit: SpectrumFit, condition: str, source: str = "") -> list[dict]:
    rows = []
    for item in fit.channels:
        peak = item.dominant_peak
        rows.append({
            "condition": CONDITION_LABELS.get(condition, condition),
            "source": source,
            "channel": item.name,
            "t_start_s": fit.t_start,
            "t_end_s": fit.t_end,
            "fmin_hz": fit.fmin,
            "fmax_hz": fit.fmax,
            "welch_segment_s": fit.segment_s,
            "welch_nperseg": fit.welch_nperseg,
            "detrend": fit.detrend,
            "aperiodic_mode": fit.aperiodic_mode,
            "usable": item.usable,
            "offset": item.offset,
            "knee": item.knee,
            "exponent": item.exponent,
            "peak_cf_hz": None if peak is None else peak[0],
            "peak_power": None if peak is None else peak[1],
            "bandwidth_hz": None if peak is None else peak[2],
            "peaks_hz": ", ".join(f"{cf:.2f}" for cf, _pw, _bw in item.peaks),
            "r_squared": item.r_squared,
            "error": item.error,
            "n_epochs": item.n_epochs,
            "n_peak_epochs": item.n_peak_epochs,
            "peak_at_width_floor": item.peak_at_width_floor,
            "note": item.reason,
        })
    return rows


def peak_rows(fit: SpectrumFit, condition: str, source: str = "") -> list[dict]:
    """One row per fitted peak, with center, height, and bandwidth."""
    label = CONDITION_LABELS.get(condition, condition)
    rows = []
    for item in fit.channels:
        for index, (center, height, width) in enumerate(item.peaks, start=1):
            rows.append({
                "condition": label,
                "source": source,
                "channel": item.name,
                "peak": index,
                "center_frequency_hz": center,
                "peak_height_log10": height,
                "bandwidth_hz": width,
                "above_aperiodic": True,
            })
    return rows


def comparison_rows(
    rows: list[ChannelComparison],
    band: tuple[float, float] | None = None,
) -> list[dict]:
    """Write the comparison. The alpha_* columns are the peak inside `band`."""
    low, high = (8.0, 12.0) if band is None else (float(band[0]), float(band[1]))
    name = band_label(low, high)
    exported = []
    for row in rows:
        exported.append({
            "channel": row.name,
            "band": name,
            "band_low_hz": low,
            "band_high_hz": high,
            "offset_ec": row.offset_ec,
            "offset_eo": row.offset_eo,
            "exponent_ec": row.exponent_ec,
            "exponent_eo": row.exponent_eo,
            "exponent_eo_minus_ec": row.exponent_delta,
            "alpha_cf_ec": row.alpha_cf_ec,
            "alpha_cf_eo": row.alpha_cf_eo,
            "alpha_power_ec": row.alpha_power_ec,
            "alpha_power_eo": row.alpha_power_eo,
            "alpha_reactivity_ec_minus_eo": row.alpha_reactivity,
            "peak_height_definition": "aperiodic-adjusted peak height, log10 power, EC minus EO",
            "peak_center_difference_hz": row.peak_center_difference_hz,
            "band_power_ec": row.band_power_ec,
            "band_power_eo": row.band_power_eo,
            "band_power_reactivity_percent": row.band_power_reactivity_percent,
            "band_power_definition": "mean Welch power in the band, (EC - EO) / EO * 100",
            "r_squared_ec": row.r_squared_ec,
            "r_squared_eo": row.r_squared_eo,
            "nearest_cf_ec": row.nearest_cf_ec,
            "nearest_distance_ec_hz": row.nearest_distance_ec,
            "nearest_cf_eo": row.nearest_cf_eo,
            "nearest_distance_eo_hz": row.nearest_distance_eo,
            "peak_at_width_floor_ec": row.peak_at_floor_ec,
            "peak_at_width_floor_eo": row.peak_at_floor_eo,
        })
    return exported


def cycle_summary_row(fit: CycleFit, condition: str, source: str = "") -> dict:
    row = {
        "condition": CONDITION_LABELS.get(condition, condition),
        "source": source,
        "channel": fit.channel,
        "band_low_hz": fit.band[0],
        "band_high_hz": fit.band[1],
        "n_cycles": fit.n_cycles,
        "n_burst_cycles": fit.n_burst_cycles,
        "n_burst_episodes": fit.n_burst_episodes,
        "burst_duration_s": fit.burst_duration_s,
        "burst_fraction": fit.burst_fraction,
        "burst_note": fit.burst_note,
        "period_ms": fit.period_ms,
        "rise_decay": fit.rise_decay,
        "peak_trough": fit.peak_trough,
        "volt_amp": fit.volt_amp,
        "burst_period_ms": fit.burst_period_ms,
        "burst_rise_decay": fit.burst_rise_decay,
        "burst_peak_trough": fit.burst_peak_trough,
        "burst_volt_amp": fit.burst_volt_amp,
    }
    for cutoff in fit.cutoffs:
        row[f"{cutoff.key}_cutoff"] = cutoff.cutoff
        row[f"{cutoff.key}_median"] = cutoff.median
        row[f"{cutoff.key}_n_passed"] = cutoff.n_passed
    return row


def burst_episode_rows(fit: CycleFit, condition: str, source: str = "") -> list[dict]:
    """One row per burst episode: when it started, how long it ran, and its median cycle."""
    label = CONDITION_LABELS.get(condition, condition)
    return [
        {
            "condition": label,
            "source": source,
            "channel": fit.channel,
            "band_low_hz": fit.band[0],
            "band_high_hz": fit.band[1],
            "episode": index + 1,
            "start_s": episode.start_s,
            "end_s": episode.end_s,
            "duration_s": episode.end_s - episode.start_s,
            "n_cycles": episode.n_cycles,
            "period_ms": episode.period_ms,
            "volt_amp": episode.volt_amp,
        }
        for index, episode in enumerate(fit.episodes)
    ]


def cycle_detail_rows(fit: CycleFit, condition: str) -> list[dict]:
    label = CONDITION_LABELS.get(condition, condition)
    return [
        {"condition": label, "channel": fit.channel, "cycle": index + 1, **row}
        for index, row in enumerate(fit.cycles)
    ]


def write_json(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def fit_metadata(
    recording: Recording,
    settings: FitSettings,
    fit: SpectrumFit | None = None,
    condition: str = "",
    extras: dict | None = None,
) -> dict:
    """Settings and exclusions behind one fit, so a table can be read later."""
    invalid = 0 if recording.valid is None else int((~recording.valid).sum())
    accepted = recording.n_samples if recording.valid is None else int(recording.valid.any(axis=0).sum())
    payload = {
        "bifooof": __version__,
        "specparam": _package_version("specparam"),
        "bycycle": _package_version("bycycle"),
        "mne": _package_version("mne"),
        "numpy": _package_version("numpy"),
        "condition": CONDITION_LABELS.get(condition, condition),
        "source": recording.source_path,
        "format": recording.source_format,
        "sfreq_hz": recording.sfreq,
        "units": "microvolts",
        "reference": "as stored in the file; BiFooof does not re-reference",
        "time_window_start_s": None if fit is None else fit.t_start,
        "time_window_end_s": None if fit is None else fit.t_end,
        "fmin_hz": settings.fmin if fit is None else fit.fmin,
        "fmax_hz": settings.fmax if fit is None else fit.fmax,
        "welch_segment_s": settings.segment_s if fit is None else fit.segment_s,
        "welch_nperseg": None if fit is None else fit.welch_nperseg,
        "detrend": "linear",
        "aperiodic_mode": settings.aperiodic_mode,
        "peak_width_hz": [settings.peak_width_low, settings.peak_width_high],
        "max_n_peaks": settings.max_n_peaks,
        "min_peak_height": settings.min_peak_height,
        "peak_threshold_sd": settings.peak_threshold,
        "band_hz": [settings.alpha_low, settings.alpha_high],
        "reference": settings.reference,
        "amplitude_limit_uv": settings.amplitude_limit_uv,
        "epoch_s": settings.epoch_s,
        "burst_thresholds": settings.cycle_thresholds(),
        "bad_channels": list(recording.bad_channels),
        "rejected_spans_s": [
            {"start": start, "end": stop} for start, stop in recording.rejected_spans
        ],
        "invalid_channel_samples": invalid,
        "accepted_duration_s": accepted / recording.sfreq if recording.sfreq else 0.0,
        "notes": list(recording.notes),
        "peak_height_definition": (
            "EC minus EO fitted peak height above the aperiodic background, in log10 power. "
            "Each condition picks its own tallest peak in the band. A missing peak stays missing."
        ),
        "band_power_definition": (
            "Mean linear Welch power in the band, in microvolts squared per Hz. "
            "The percent is the change between recordings, (EC - EO) / EO * 100. "
            "It is not the band's share of total spectral power. "
            "It is blank when eyes-open power is missing or not positive."
        ),
        "epoch_peak_definition": (
            "When epochs are used, each epoch is fit alone. "
            "Peaks within 1 Hz are grouped. Center, height, and width are medians of the epochs in the group. "
            "An epoch that missed the peak is omitted, not entered as height zero. "
            "n_peak_epochs counts epochs with an in-band peak and does not scale the height."
        ),
        "burst_definition": (
            "n_burst_cycles counts cycles labelled as part of a burst. "
            "n_burst_episodes counts contiguous runs of those cycles. "
            "A run cannot cross a rejected interval. "
            "A cycle clears a cutoff only when its value is strictly above it. "
            "The first and last cycle of a fitted run are never bursts, "
            "and a run shorter than min_n_cycles is dropped."
        ),
    }
    if extras:
        payload.update(extras)
    return payload


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", ""))


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
