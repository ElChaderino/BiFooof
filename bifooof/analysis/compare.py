"""Eyes-closed versus eyes-open spectral comparison."""

from __future__ import annotations

from dataclasses import dataclass

from .specparam_fit import ChannelFit, SpectrumFit

ALPHA = (8.0, 12.0)


@dataclass
class ChannelComparison:
    name: str
    offset_ec: float | None = None
    offset_eo: float | None = None
    exponent_ec: float | None = None
    exponent_eo: float | None = None
    exponent_delta: float | None = None
    alpha_cf_ec: float | None = None
    alpha_cf_eo: float | None = None
    alpha_power_ec: float | None = None
    alpha_power_eo: float | None = None
    alpha_reactivity: float | None = None


def alpha_peak(
    channel: ChannelFit | None,
    band: tuple[float, float] | None = None,
) -> tuple[float, float, float] | None:
    """Highest peak whose center sits in the alpha band. The default band is 8–12 Hz."""
    if channel is None or not channel.peaks:
        return None
    low, high = ALPHA if band is None else band
    chosen = [peak for peak in channel.peaks if low <= peak[0] <= high]
    if not chosen:
        return None
    return max(chosen, key=lambda peak: peak[1])


def compare_spectra(
    eyes_closed: SpectrumFit,
    eyes_open: SpectrumFit,
    alpha: tuple[float, float] | None = None,
) -> list[ChannelComparison]:
    """Match channels by name and subtract eyes-open from eyes-closed alpha power."""
    band = ALPHA if alpha is None else alpha
    closed = {item.name: item for item in eyes_closed.channels}
    opened = {item.name: item for item in eyes_open.channels}
    rows = []
    for name in _order([*closed.keys(), *opened.keys()]):
        ec = closed.get(name)
        eo = opened.get(name)
        ec_alpha = alpha_peak(ec, band)
        eo_alpha = alpha_peak(eo, band)
        exponent_ec = None if ec is None else ec.exponent
        exponent_eo = None if eo is None else eo.exponent
        power_ec = None if ec_alpha is None else ec_alpha[1]
        power_eo = None if eo_alpha is None else eo_alpha[1]
        rows.append(ChannelComparison(
            name=name,
            offset_ec=None if ec is None else ec.offset,
            offset_eo=None if eo is None else eo.offset,
            exponent_ec=exponent_ec,
            exponent_eo=exponent_eo,
            exponent_delta=_sub(exponent_eo, exponent_ec),
            alpha_cf_ec=None if ec_alpha is None else ec_alpha[0],
            alpha_cf_eo=None if eo_alpha is None else eo_alpha[0],
            alpha_power_ec=power_ec,
            alpha_power_eo=power_eo,
            alpha_reactivity=_sub(power_ec, power_eo),
        ))
    return rows


def _order(names: list[str]) -> list[str]:
    from ..io.channels import STANDARD_19

    seen: list[str] = []
    for name in list(STANDARD_19) + names:
        if name in names and name not in seen:
            seen.append(name)
    return seen


def _sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)
