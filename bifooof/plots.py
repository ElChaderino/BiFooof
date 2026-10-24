"""Matplotlib figures shared by the window and the command line."""

from __future__ import annotations

import matplotlib
from matplotlib.ticker import NullFormatter

matplotlib.rcParams.update({
    "axes.edgecolor": "#d2dbe6",
    "axes.labelcolor": "#16232f",
    "axes.titlecolor": "#16232f",
    "xtick.color": "#63748a",
    "ytick.color": "#63748a",
    "text.color": "#16232f",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
})

import numpy as np

from .analysis.compare import alpha_peak, compare_spectra
from .analysis.prepare import R2_WEAK
from .analysis.cycles import CycleFit
from .analysis.settings import band_phrase
from .analysis.specparam_fit import SpectrumFit, log_residual
from .ui.headmap import draw_head

_EC = "#1c4e78"
_EO = "#c2410c"
_BANDS = (
    (1.0, 4.0, "#f4f7fb"),
    (4.0, 8.0, "#eef6f5"),
    (8.0, 12.0, "#e8f0f8"),
    (13.0, 30.0, "#fbf6f1"),
    (30.0, 45.0, "#f7f4fb"),
)


def mark_spectrum(ax, fmin: float, fmax: float) -> None:
    """Shade the usual bands and label a few frequencies on a log axis."""
    ax.set_xscale("log")
    for low, high, color in _BANDS:
        left = max(low, fmin)
        right = min(high, fmax)
        if right > left + 0.2:
            ax.axvspan(left, right, color=color, zorder=0, lw=0)
    ticks = [tick for tick in (1, 2, 4, 8, 10, 13, 20, 30, 40, 45) if fmin - 0.05 <= tick <= fmax + 0.05]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(fmin, fmax)


def stamp_figure(fig) -> None:
    """Saved figures stay unmarked. The argument is kept for existing callers."""
    del fig


def finish_figure(fig) -> None:
    """Space axes so titles and colorbars do not collide."""
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.tight_layout()
        except ValueError:
            return


def clear_figure(fig) -> None:
    """Remove the axes. A log spectrum axis rejects the limits used while clearing."""
    for ax in list(fig.axes):
        try:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        except (ValueError, RuntimeError):
            continue
    fig.clear()


def style_axes(ax, grid: bool = False) -> None:
    """Apply the clinical axis chrome used by every plot."""
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d2dbe6")
    ax.spines["bottom"].set_color("#d2dbe6")
    ax.tick_params(colors="#63748a", labelsize=8)
    ax.xaxis.label.set_color("#16232f")
    ax.yaxis.label.set_color("#16232f")
    ax.title.set_color("#16232f")
    if grid:
        ax.grid(True, which="both", color="#e4eaf1", lw=0.6)
        ax.set_axisbelow(True)


def render_spectrum(fig, fit: SpectrumFit, index: int, label: str = "", mark_hz: float | None = None) -> None:
    """Draw one condition's spectrum, the residual, and the head maps onto ``fig``."""
    if not fit.channels:
        return
    chosen = min(max(index, 0), len(fit.channels) - 1)
    item = fit.channels[chosen]
    clear_figure(fig)
    finish_figure(fig)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06)
    curves = outer[0, 0].subgridspec(4, 1, height_ratios=[1.25, 1.2, 1.1, 0.72], hspace=0.06)
    heads = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
    spectrum = fig.add_subplot(curves[:3, 0])
    spectrum.set_label("spectrum")
    residual = fig.add_subplot(curves[3, 0], sharex=spectrum)
    residual.set_label("residual")
    if item.freqs is None or item.spectrum is None:
        spectrum.axis("off")
        spectrum.text(0.5, 0.5, item.reason or "No fit for this channel.",
                      ha="center", va="center", color="#63748a")
        residual.axis("off")
    else:
        mark_spectrum(spectrum, fit.fmin, fit.fmax)
        spectrum.plot(item.freqs, item.spectrum, color="#16324f", lw=1.7, label="Spectrum", zorder=2)
        if item.model is not None:
            spectrum.plot(item.freqs, item.model, color="#c2410c", lw=2.0, alpha=0.95, label="Model", zorder=3)
        if item.aperiodic is not None:
            spectrum.plot(item.freqs, item.aperiodic, color="#0e8f86", lw=1.5, ls="--", label="Aperiodic", zorder=3)
        _mark_peaks(spectrum, item)
        spectrum.set_xlabel("")
        spectrum.tick_params(labelbottom=False)
        spectrum.set_ylabel("log10 power")
        title = (
            f"{item.name}   ·   {fit.t_start:.0f}–{fit.t_end:.0f} s"
            f"   ·   {fit.fmin:g}–{fit.fmax:.0f} Hz"
        )
        if label:
            title = f"{label}   ·   {title}"
        spectrum.set_title(title, fontsize=10, loc="left")
        spectrum.legend(frameon=False, fontsize=8)
        style_axes(spectrum, grid=True)
        _draw_residual(residual, item, mark_hz)
        if mark_hz is not None:
            spectrum.axvline(mark_hz, color="#63748a", lw=0.7, ls="--", zorder=1)

    exponents = [channel.exponent for channel in fit.channels]
    centers = [channel.dominant_peak[0] if channel.dominant_peak else None for channel in fit.channels]
    names = [channel.name for channel in fit.channels]
    weak = [channel.r_squared is not None and channel.r_squared < R2_WEAK for channel in fit.channels]
    floors = [weak[index] or channel.peak_at_width_floor for index, channel in enumerate(fit.channels)]
    draw_head(fig.add_subplot(heads[0, 0]), names, exponents, f"Exponent  {fit.fmin:g}–{fit.fmax:.0f} Hz", "viridis", item.name, weak)
    draw_head(fig.add_subplot(heads[1, 0]), names, centers, "Peak frequency (Hz)", "plasma", item.name, floors)
    stamp_figure(fig)


def _draw_residual(ax, item, mark_hz: float | None) -> None:
    curves = log_residual(item)
    if curves is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No model residual.", ha="center", va="center", color="#63748a", fontsize=8)
        return
    freqs, values = curves
    ax.axhline(0.0, color="#d2dbe6", lw=0.8, zorder=1)
    ax.plot(freqs, values, color="#16324f", lw=1.05, zorder=2)
    if mark_hz is not None:
        ax.axvline(mark_hz, color="#63748a", lw=0.7, ls="--", zorder=1)
    ax.set_ylabel("log10 residual")
    ax.set_xlabel("Frequency (Hz)")
    ax.tick_params(labelbottom=True)
    style_axes(ax, grid=True)


def render_comparison(fig, eyes_closed: SpectrumFit, eyes_open: SpectrumFit, channel: str,
                      alpha: tuple[float, float] = (8.0, 12.0), peak_width_low: float | None = None):
    """Overlay eyes-closed and eyes-open spectra. Returns the comparison rows."""
    rows = compare_spectra(eyes_closed, eyes_open, alpha, peak_width_low)
    clear_figure(fig)
    finish_figure(fig)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06)
    curves = outer[0, 0].subgridspec(4, 1, height_ratios=[1.25, 1.2, 1.1, 0.72], hspace=0.06)
    maps = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
    ax = fig.add_subplot(curves[:3, 0])
    fmin = min(eyes_closed.fmin, eyes_open.fmin)
    fmax = max(eyes_closed.fmax, eyes_open.fmax)
    mark_spectrum(ax, fmin, fmax)
    _plot_condition(ax, _by_name(eyes_closed, channel), _EC, "Eyes closed", alpha)
    _plot_condition(ax, _by_name(eyes_open, channel), _EO, "Eyes open", alpha)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10 power")
    shown = band_phrase(*alpha)
    ax.set_title(f"{channel}   ·   {shown}\neyes closed vs eyes open", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, grid=True)
    names = [row.name for row in rows]
    draw_head(
        fig.add_subplot(maps[0, 0]), names,
        [row.exponent_delta for row in rows],
        f"Exponent\n{fmin:g}–{fmax:.0f} Hz  ·  EO − EC", "viridis", channel,
        [_weak_fit(row.r_squared_ec) or _weak_fit(row.r_squared_eo) for row in rows],
    )
    draw_head(
        fig.add_subplot(maps[1, 0]), names,
        [row.alpha_reactivity for row in rows],
        f"{shown}\nPeak height  ·  EC − EO", "plasma", channel,
        [
            _weak_fit(row.r_squared_ec) or _weak_fit(row.r_squared_eo) or row.peak_at_floor_ec or row.peak_at_floor_eo
            for row in rows
        ],
    )
    _peak_lines(fig.add_subplot(curves[3, 0]), rows)
    stamp_figure(fig)
    return rows


def _peak_lines(ax, rows) -> None:
    """Saved comparison uses the same slopegraph as the Compare page."""
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Eyes closed", "Eyes open"])
    drawn = 0
    for row in rows:
        pair = []
        xpos = []
        if row.alpha_power_ec is not None and np.isfinite(row.alpha_power_ec):
            pair.append(float(row.alpha_power_ec))
            xpos.append(0)
        if row.alpha_power_eo is not None and np.isfinite(row.alpha_power_eo):
            pair.append(float(row.alpha_power_eo))
            xpos.append(1)
        if not pair:
            continue
        if len(pair) == 2:
            ax.plot(xpos, pair, color="#9aafc4", lw=0.7, zorder=1)
            drawn += 1
        ax.scatter([xpos[0]], [pair[0]], s=16, color=_EC if xpos[0] == 0 else _EO, zorder=2, linewidths=0)
        if len(pair) == 2:
            ax.scatter([1], [pair[1]], s=16, color=_EO, zorder=2, linewidths=0)
    ax.set_ylabel("Peak height")
    ax.set_title(
        f"{drawn} channels with paired peaks\nA missing peak is a single dot",
        fontsize=9, loc="left",
    )
    style_axes(ax, grid=True)


def render_cycles(fig, fit: CycleFit, seconds: float = 10.0) -> None:
    """Draw the first page of a bycycle fit."""
    clear_figure(fig)
    ax = fig.add_subplot(111)
    if fit.filtered.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No samples in this cycle fit.", ha="center", va="center", color="#63748a")
        return
    stop = min(fit.filtered.size, max(1, int(round(seconds * fit.sfreq))))
    times = fit.t_offset + np.arange(stop) / fit.sfreq
    step = max(1, stop // 2500)
    for span_start, span_stop, is_burst in fit.spans:
        if not is_burst or span_start >= stop:
            continue
        left = fit.t_offset + span_start / fit.sfreq
        right = fit.t_offset + min(span_stop, stop) / fit.sfreq
        ax.axvspan(left, right, color="#0e8f86", alpha=0.28, lw=0, zorder=0)
    peaks = [peak for peak in fit.peaks if 0 <= peak < stop]
    if peaks:
        ax.plot(fit.t_offset + np.asarray(peaks) / fit.sfreq, fit.filtered[peaks],
                linestyle="none", marker="o", ms=3.5, color="#b3261e", zorder=3)
    ax.plot(times[::step], fit.filtered[:stop][::step], color="#16324f", lw=0.9, zorder=2)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    ax.set_title(
        f"{fit.channel} filtered {fit.band[0]:g}–{fit.band[1]:g} Hz"
        f"    ·    {fit.n_cycles} cycles, {fit.n_burst_cycles} burst cycles "
        f"in {fit.n_burst_episodes} episodes"
        + (f"    ·    {fit.burst_duration_s:.2f} s" if fit.n_burst_episodes else ""),
        fontsize=9, loc="left",
    )
    style_axes(ax)
    fig.tight_layout()
    stamp_figure(fig)


def _weak_fit(value: float | None) -> bool:
    return value is not None and value < R2_WEAK


def _by_name(fit: SpectrumFit, name: str):
    for item in fit.channels:
        if item.name == name:
            return item
    return None


def _mark_peaks(ax, item) -> None:
    if item.aperiodic is None or not item.peaks:
        return
    dominant = item.dominant_peak
    for center, power, _width in item.peaks:
        base = float(np.interp(center, item.freqs, item.aperiodic))
        ax.plot(center, base + power, "o", color="#b3261e", ms=5, zorder=4)
    if dominant is None:
        return
    base = float(np.interp(dominant[0], item.freqs, item.aperiodic))
    ax.annotate(
        f"{dominant[0]:.1f} Hz",
        (dominant[0], base + dominant[1]),
        textcoords="offset points", xytext=(6, 6),
        fontsize=8, color="#b3261e",
    )


def _plot_condition(ax, item, color: str, label: str, alpha: tuple[float, float] = (8.0, 12.0)) -> None:
    if item is None or item.freqs is None or item.spectrum is None:
        return
    ax.plot(item.freqs, item.spectrum, color=color, lw=1.8, label=label)
    if item.aperiodic is not None:
        ax.plot(item.freqs, item.aperiodic, color=color, lw=1.2, ls="--", alpha=0.85)
    peak = alpha_peak(item, alpha)
    if peak is not None and item.aperiodic is not None:
        base = float(np.interp(peak[0], item.freqs, item.aperiodic))
        ax.plot(peak[0], base + peak[1], "o", color=color, ms=6)
