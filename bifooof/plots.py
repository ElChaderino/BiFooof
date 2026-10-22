"""Matplotlib figures shared by the window and the command line."""

from __future__ import annotations

import matplotlib

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
        spectrum.plot(item.freqs, item.spectrum, color="#16324f", lw=1.6, label="Spectrum")
        if item.model is not None:
            spectrum.plot(item.freqs, item.model, color="#c2410c", lw=2.0, alpha=0.9, label="Model")
        if item.aperiodic is not None:
            spectrum.plot(item.freqs, item.aperiodic, color="#0e8f86", lw=1.6, ls="--", label="Aperiodic")
        for center, power, _width in item.peaks:
            if item.aperiodic is None:
                continue
            base = float(np.interp(center, item.freqs, item.aperiodic))
            spectrum.plot(center, base + power, "o", color="#b3261e", ms=5)
        spectrum.set_xscale("log")
        spectrum.set_xlabel("")
        spectrum.set_ylabel("log10 power")
        title = f"{item.name}   ·   {fit.fmin:g}–{fit.fmax:.0f} Hz   ·   frequency"
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


def render_comparison(fig, eyes_closed: SpectrumFit, eyes_open: SpectrumFit, channel: str):
    """Overlay eyes-closed and eyes-open spectra. Returns the comparison rows."""
    rows = compare_spectra(eyes_closed, eyes_open)
    fig.clear()
    finish_figure(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.45])
    ax = fig.add_subplot(grid[0, :])
    _plot_condition(ax, _by_name(eyes_closed, channel), _EC, "Eyes closed")
    _plot_condition(ax, _by_name(eyes_open, channel), _EO, "Eyes open")
    ax.set_xscale("log")
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
        "Alpha power  EC − EO", "plasma", channel,
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
    times = np.arange(stop) / fit.sfreq
    step = max(1, stop // 2500)
    for span_start, span_stop, is_burst in fit.spans:
        if span_start >= stop:
            break
        left = span_start / fit.sfreq
        right = min(span_stop, stop) / fit.sfreq
        ax.axvspan(left, right, color="#0e8f86" if is_burst else "#d5dde6", alpha=0.45, lw=0)
    ax.plot(times[::step], fit.filtered[:stop][::step], color="#16324f", lw=0.9)
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


def _plot_condition(ax, item, color: str, label: str) -> None:
    if item is None or item.freqs is None or item.spectrum is None:
        return
    ax.plot(item.freqs, item.spectrum, color=color, lw=1.8, label=label)
    if item.aperiodic is not None:
        ax.plot(item.freqs, item.aperiodic, color=color, lw=1.2, ls="--", alpha=0.85)
    peak = alpha_peak(item)
    if peak is not None and item.aperiodic is not None:
        base = float(np.interp(peak[0], item.freqs, item.aperiodic))
        ax.plot(peak[0], base + peak[1], "o", color=color, ms=6)
