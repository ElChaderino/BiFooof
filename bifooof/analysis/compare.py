"""Eyes-closed versus eyes-open spectral comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prepare import peak_at_width_floor
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
    band_power_ec: float | None = None
    band_power_eo: float | None = None
    band_power_reactivity_percent: float | None = None
    peak_center_difference_hz: float | None = None
    r_squared_ec: float | None = None
    r_squared_eo: float | None = None
    nearest_cf_ec: float | None = None
    nearest_distance_ec: float | None = None
    nearest_cf_eo: float | None = None
    nearest_distance_eo: float | None = None
    peak_at_floor_ec: bool = False
    peak_at_floor_eo: bool = False


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


def nearest_outside_peak(
    channel: ChannelFit | None,
    band: tuple[float, float],
) -> tuple[float, float] | None:
    """Closest peak center outside the band, and how far outside it sits.

    Empty when the channel already has a peak inside the band.
    """
    if channel is None or not channel.peaks or alpha_peak(channel, band) is not None:
        return None
    low, high = band

    def distance(center: float) -> float:
        if center < low:
            return low - center
        if center > high:
            return center - high
        return 0.0

    center = min(channel.peaks, key=lambda peak: (distance(peak[0]), -peak[1]))[0]
    gap = distance(center)
    if gap <= 0:
        return None
    return float(center), float(gap)


def band_power(channel: ChannelFit | None, band: tuple[float, float]) -> float | None:
    """Mean linear power in ``band``, in microvolts squared per Hz.

    This is the Welch spectrum inside the band, not the fitted peak height.
    """
    if channel is None or channel.freqs is None or channel.spectrum is None:
        return None
    low, high = band
    chosen = (channel.freqs >= low) & (channel.freqs <= high) & np.isfinite(channel.spectrum)
    if not np.any(chosen):
        return None
    linear = np.power(10.0, np.asarray(channel.spectrum[chosen], dtype=np.float64))
    return float(np.mean(linear))


BAND_CHANGE = (
    "Percent change between the two recordings in mean Welch power inside the band: "
    "(eyes closed − eyes open) / eyes open × 100. "
    "It is not the band's share of total spectral power. "
    "It stays blank when eyes-open power is missing or not positive."
)


def band_reactivity_percent(eyes_closed: float | None, eyes_open: float | None) -> float | None:
    """``(EC − EO) / EO × 100``. Missing when either side is missing or eyes-open power is 0."""
    if (
        eyes_closed is None or eyes_open is None
        or not np.isfinite(eyes_closed) or not np.isfinite(eyes_open)
        or eyes_open <= 0
    ):
        return None
    return float((eyes_closed - eyes_open) / eyes_open * 100.0)


def compare_pair(
    left: SpectrumFit,
    right: SpectrumFit,
    band: tuple[float, float] | None = None,
    peak_width_low: float | None = None,
) -> list[ChannelComparison]:
    """Match two fits by channel name.

    Left-hand values are stored in the ``_ec`` fields and right-hand values in
    the ``_eo`` fields. Eyes closed versus eyes open is the usual assignment.
    Pre/post uses the same numbers with different labels on screen.

    ``alpha_reactivity`` is left peak height minus right peak height. Each side
    keeps its own tallest peak in the band. A missing peak stays missing.
    """
    return _compare(left, right, ALPHA if band is None else band, peak_width_low)


def compare_spectra(
    eyes_closed: SpectrumFit,
    eyes_open: SpectrumFit,
    alpha: tuple[float, float] | None = None,
    peak_width_low: float | None = None,
) -> list[ChannelComparison]:
    """Eyes closed against eyes open. See :func:`compare_pair`."""
    return compare_pair(eyes_closed, eyes_open, alpha, peak_width_low)


def _compare(
    eyes_closed: SpectrumFit,
    eyes_open: SpectrumFit,
    band: tuple[float, float],
    peak_width_low: float | None = None,
) -> list[ChannelComparison]:
    """Match channels by name.

    ``alpha_reactivity`` is the eyes-closed fitted peak height minus the
    eyes-open fitted peak height. Each side keeps its own tallest peak in the
    band, so the two centers need not be the same component. A missing peak
    stays missing; it is not treated as zero.

    ``band_power_reactivity_percent`` is the separate, conventional change in
    mean Welch power across the same band.
    """
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
        cf_ec = None if ec_alpha is None else ec_alpha[0]
        cf_eo = None if eo_alpha is None else eo_alpha[0]
        band_ec = band_power(ec, band)
        band_eo = band_power(eo, band)
        nearest_ec = nearest_outside_peak(ec, band)
        nearest_eo = nearest_outside_peak(eo, band)
        rows.append(ChannelComparison(
            name=name,
            offset_ec=None if ec is None else ec.offset,
            offset_eo=None if eo is None else eo.offset,
            exponent_ec=exponent_ec,
            exponent_eo=exponent_eo,
            exponent_delta=_sub(exponent_eo, exponent_ec),
            alpha_cf_ec=cf_ec,
            alpha_cf_eo=cf_eo,
            alpha_power_ec=power_ec,
            alpha_power_eo=power_eo,
            alpha_reactivity=_sub(power_ec, power_eo),
            band_power_ec=band_ec,
            band_power_eo=band_eo,
            band_power_reactivity_percent=band_reactivity_percent(band_ec, band_eo),
            peak_center_difference_hz=None if cf_ec is None or cf_eo is None else abs(cf_ec - cf_eo),
            r_squared_ec=None if ec is None else ec.r_squared,
            r_squared_eo=None if eo is None else eo.r_squared,
            nearest_cf_ec=None if nearest_ec is None else nearest_ec[0],
            nearest_distance_ec=None if nearest_ec is None else nearest_ec[1],
            nearest_cf_eo=None if nearest_eo is None else nearest_eo[0],
            nearest_distance_eo=None if nearest_eo is None else nearest_eo[1],
            peak_at_floor_ec=_at_floor(ec, ec_alpha, peak_width_low),
            peak_at_floor_eo=_at_floor(eo, eo_alpha, peak_width_low),
        ))
    return rows


def _at_floor(channel: ChannelFit | None, peak: tuple[float, float, float] | None, width_low: float | None) -> bool:
    """The floor flag follows the peak in the current band, not the band used at fit time."""
    if channel is None or peak is None:
        return False
    if width_low is None:
        return bool(channel.peak_at_width_floor)
    return peak_at_width_floor(peak[2], width_low)


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
