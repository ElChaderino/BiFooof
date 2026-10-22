"""Save plots and fit tables."""

from __future__ import annotations

import csv
from pathlib import Path

from .analysis.compare import ChannelComparison
from .analysis.cycles import CycleFit
from .analysis.settings import band_label
from .analysis.specparam_fit import SpectrumFit
from .io.condition import CONDITION_LABELS


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
            "note": item.reason,
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
        })
    return exported


def cycle_summary_row(fit: CycleFit, condition: str, source: str = "") -> dict:
    return {
        "condition": CONDITION_LABELS.get(condition, condition),
        "source": source,
        "channel": fit.channel,
        "band_low_hz": fit.band[0],
        "band_high_hz": fit.band[1],
        "n_cycles": fit.n_cycles,
        "n_bursts": fit.n_bursts,
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


def cycle_detail_rows(fit: CycleFit, condition: str) -> list[dict]:
    label = CONDITION_LABELS.get(condition, condition)
    return [
        {"condition": label, "channel": fit.channel, "cycle": index + 1, **row}
        for index, row in enumerate(fit.cycles)
    ]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
