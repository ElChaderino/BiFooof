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
from .analysis.cycles import CycleFit
from .analysis.specparam_fit import SpectrumFit
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


def finish_figure(fig) -> None:
    """Space axes so titles and colorbars do not collide."""
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.tight_layout()
        except ValueError:
            return


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


def render_spectrum(fig, fit: SpectrumFit, index: int, label: str = "") -> None:
    """Draw one condition's spectrum, model, and head maps onto ``fig``."""
    if not fit.channels:
        return
    chosen = min(max(index, 0), len(fit.channels) - 1)
    item = fit.channels[chosen]
    fig.clear()
    finish_figure(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.45])
    spectrum = fig.add_subplot(grid[0, :])
    if item.freqs is None or item.spectrum is None:
        spectrum.axis("off")
        spectrum.text(0.5, 0.5, item.reason or "No fit for this channel.",
                      ha="center", va="center", color="#63748a")
    else:
        mark_spectrum(spectrum, fit.fmin, fit.fmax)
        spectrum.plot(item.freqs, item.spectrum, color="#16324f", lw=1.7, label="Spectrum", zorder=2)
        if item.model is not None:
            spectrum.plot(item.freqs, item.model, color="#c2410c", lw=2.0, alpha=0.95, label="Model", zorder=3)
        if item.aperiodic is not None:
            spectrum.plot(item.freqs, item.aperiodic, color="#0e8f86", lw=1.5, ls="--", label="Aperiodic", zorder=3)
        _mark_peaks(spectrum, item)
        spectrum.set_xlabel("")
        spectrum.set_ylabel("log10 power")
        title = (
            f"{item.name}   ·   {fit.t_start:.0f}–{fit.t_end:.0f} s"
            f"   ·   {fit.fmin:g}–{fit.fmax:.0f} Hz   ·   frequency"
        )
        if label:
            title = f"{label}   ·   {title}"
        spectrum.set_title(title, fontsize=10, loc="left")
        spectrum.legend(frameon=False, fontsize=8)
        style_axes(spectrum, grid=True)

    exponents = [channel.exponent for channel in fit.channels]
    centers = [channel.dominant_peak[0] if channel.dominant_peak else None for channel in fit.channels]
    names = [channel.name for channel in fit.channels]
    draw_head(fig.add_subplot(grid[1, 0]), names, exponents, "Exponent", "viridis", item.name)
    draw_head(fig.add_subplot(grid[1, 1]), names, centers, "Peak frequency (Hz)", "plasma", item.name)


def render_comparison(fig, eyes_closed: SpectrumFit, eyes_open: SpectrumFit, channel: str,
                      alpha: tuple[float, float] = (8.0, 12.0)):
    """Overlay eyes-closed and eyes-open spectra. Returns the comparison rows."""
    rows = compare_spectra(eyes_closed, eyes_open, alpha)
    fig.clear()
    finish_figure(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.45])
    ax = fig.add_subplot(grid[0, :])
    fmin = min(eyes_closed.fmin, eyes_open.fmin)
    fmax = max(eyes_closed.fmax, eyes_open.fmax)
    mark_spectrum(ax, fmin, fmax)
    _plot_condition(ax, _by_name(eyes_closed, channel), _EC, "Eyes closed", alpha)
    _plot_condition(ax, _by_name(eyes_open, channel), _EO, "Eyes open", alpha)
    ax.set_xlabel("")
    ax.set_ylabel("log10 power")
    ax.set_title(f"{channel}   ·   eyes closed vs eyes open   ·   frequency (Hz)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, grid=True)
    names = [row.name for row in rows]
    draw_head(
        fig.add_subplot(grid[1, 0]), names,
        [row.exponent_delta for row in rows],
        "Exponent  EO − EC", "viridis", channel,
    )
    draw_head(
        fig.add_subplot(grid[1, 1]), names,
        [row.alpha_reactivity for row in rows],
        f"Alpha {alpha[0]:g}–{alpha[1]:g} Hz   EC − EO", "plasma", channel,
    )
    return rows


def render_cycles(fig, fit: CycleFit, seconds: float = 10.0) -> None:
    """Draw the first page of a bycycle fit."""
    fig.clear()
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
        f"    ·    {fit.n_cycles} cycles, {fit.n_bursts} in bursts",
        fontsize=9, loc="left",
    )
    style_axes(ax)
    fig.tight_layout()


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
