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
from .analysis.settings import band_phrase, format_hz, format_seconds, power_unit, reference_phrase
from .analysis.cycles import CycleFit
from .analysis.specparam_fit import SpectrumFit, log_residual
from .ui.headmap import draw_head

_EC = "#1c4e78"
_EO = "#c2410c"
_BANDS = (
    (1.0, 4.0, "#d9e4f2", "delta"),
    (4.0, 8.0, "#d7efe9", "theta"),
    (8.0, 12.0, "#c9dcf0", "alpha"),
    (13.0, 30.0, "#f3e4d4", "beta"),
    (30.0, 45.0, "#e6dff0", "gamma"),
)


def mark_spectrum(ax, fmin: float, fmax: float) -> None:
    """Shade the usual bands and label a few frequencies on a log axis."""
    ax.set_xscale("log")
    for low, high, color, name in _BANDS:
        left = max(low, fmin)
        right = min(high, fmax)
        if right <= left + 0.2:
            continue
        ax.axvspan(left, right, color=color, zorder=0, lw=0)
        ax.text(
            float(np.sqrt(left * right)), 0.98, name,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7, color="#63748a", clip_on=True,
        )
    ticks = [tick for tick in (1, 2, 4, 8, 10, 13, 20, 30, 40, 45) if fmin - 0.05 <= tick <= fmax + 0.05]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(fmin, fmax)


def remember_figure(fig, *parts: str) -> None:
    """Keep the line a saved PNG should carry. The on-screen figure is not stamped."""
    fig._bifooof_stamp = "   ·   ".join(part for part in parts if part)


def stamp_figure(fig) -> None:
    """On-screen figures stay unmarked. Saving reads the remembered line."""
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


def _power_label(*fits: SpectrumFit) -> str:
    """Log-power axis label, naming scaled and current-source-density units."""
    units: list[str] = []
    for fit in fits:
        applied = (fit.provenance or {}).get("reference_applied", "stored")
        label = power_unit(str(applied))
        if label not in units:
            units.append(label)
    shown = units[0] if len(units) == 1 else " and ".join(units)
    return f"log10 power ({shown})"


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
        spectrum.set_ylabel(_power_label(fit))
        title = (
            f"{item.name}   ·   {format_seconds(fit.t_start)}–{format_seconds(fit.t_end)} s"
            f"   ·   {fit.fmin:g}–{format_hz(fit.fmax)} Hz"
        )
        if label:
            title = f"{label}   ·   {title}"
        spectrum.set_title(title, fontsize=10, loc="left")
        spectrum.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(spectrum, grid=True)
        _draw_residual(residual, item, mark_hz)
        if mark_hz is not None:
            spectrum.axvline(mark_hz, color="#63748a", lw=0.7, ls="--", zorder=1)

    exponents = [channel.exponent for channel in fit.channels]
    centers = [channel.dominant_peak[0] if channel.dominant_peak else None for channel in fit.channels]
    names = [channel.name for channel in fit.channels]
    weak = [channel.r_squared is not None and channel.r_squared < R2_WEAK for channel in fit.channels]
    floors = [weak[index] or channel.peak_at_width_floor for index, channel in enumerate(fit.channels)]
    draw_head(fig.add_subplot(heads[0, 0]), names, exponents, f"Exponent  {fit.fmin:g}–{format_hz(fit.fmax)} Hz", "viridis", item.name, weak, unit="exponent")
    draw_head(fig.add_subplot(heads[1, 0]), names, centers, "Peak frequency (Hz)", "plasma", item.name, floors, unit="Hz")
    applied = str((fit.provenance or {}).get("reference_applied", "stored"))
    remember_figure(
        fig, item.name,
        f"{format_seconds(fit.t_start)}–{format_seconds(fit.t_end)} s",
        reference_phrase(applied), power_unit(applied),
    )
    stamp_figure(fig)


def _draw_residual(ax, item, mark_hz: float | None) -> None:
    curves = log_residual(item)
    if curves is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No model residual.", ha="center", va="center", color="#63748a", fontsize=8)
        return
    freqs, values = curves
    ax.axhline(0.0, color="#d2dbe6", lw=0.8, zorder=1)
    ax.fill_between(freqs, values, 0.0, color="#d9e4f2", zorder=1)
    ax.plot(freqs, values, color="#16324f", lw=1.05, zorder=2)
    reach = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.05
    span = max(reach * 1.25, 0.05)
    ax.set_ylim(-span, span)
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
    ax.set_ylabel(_power_label(eyes_closed, eyes_open))
    shown = band_phrase(*alpha)
    ax.set_title(f"{channel}   ·   {shown}\neyes closed vs eyes open", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, grid=True)
    names = [row.name for row in rows]
    draw_head(
        fig.add_subplot(maps[0, 0]), names,
        [row.exponent_delta for row in rows],
        f"Exponent\n{fmin:g}–{format_hz(fmax)} Hz  ·  EO − EC", "RdBu_r", channel,
        [_weak_fit(row.r_squared_ec) or _weak_fit(row.r_squared_eo) for row in rows],
        diverging=True, unit="EO − EC",
    )
    draw_head(
        fig.add_subplot(maps[1, 0]), names,
        [row.alpha_reactivity for row in rows],
        f"{shown}\nPeak height  ·  EC − EO", "RdBu_r", channel,
        [
            _weak_fit(row.r_squared_ec) or _weak_fit(row.r_squared_eo) or row.peak_at_floor_ec or row.peak_at_floor_eo
            for row in rows
        ],
        diverging=True, unit="EC − EO",
    )
    _peak_lines(fig.add_subplot(curves[3, 0]), rows, channel)
    applied = [
        reference_phrase(str((fit.provenance or {}).get("reference_applied", "stored")))
        for fit in (eyes_closed, eyes_open)
    ]
    remember_figure(fig, channel, shown, " · ".join(dict.fromkeys(applied)), _power_label(eyes_closed, eyes_open))
    stamp_figure(fig)
    return rows


def _peak_lines(ax, rows, selected: str | None = None) -> None:
    """Saved comparison uses the same slopegraph as the Compare page."""
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Eyes closed", "Eyes open"])
    drawn = _slope_rows(ax, rows, selected)
    ax.set_ylabel("Peak height")
    ax.set_title(
        f"{drawn} paired channels. An open marker is one side only.",
        fontsize=9, loc="left",
    )
    style_axes(ax, grid=True)


def _slope_rows(ax, rows, selected: str | None) -> int:
    """Draw every channel. The selected one stays in front. One side is an open marker."""
    drawn = 0
    ordered = sorted(rows, key=lambda row: row.name == selected)
    for row in ordered:
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
        chosen = row.name == selected
        if len(pair) == 2:
            ax.plot(
                xpos, pair, color="#1c4e78" if chosen else "#d2dbe6",
                lw=2.2 if chosen else 0.7, zorder=3 if chosen else 1,
            )
            drawn += 1
        for x_pos, height in zip(xpos, pair):
            color = _EC if x_pos == 0 else _EO
            if len(pair) == 2:
                ax.scatter(
                    [x_pos], [height], s=36 if chosen else 12, color=color,
                    zorder=4 if chosen else 2, linewidths=0,
                )
            else:
                ax.scatter(
                    [x_pos], [height], s=42 if chosen else 22, facecolors="none",
                    edgecolors=color, linewidths=1.4, zorder=4 if chosen else 2,
                )
    return drawn


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
        ax.axvspan(left, right, color="#0e8f86", alpha=0.45, lw=0, zorder=0, label="Burst")
    peaks = [peak for peak in fit.peaks if 0 <= peak < stop]
    if peaks:
        ax.plot(fit.t_offset + np.asarray(peaks) / fit.sfreq, fit.filtered[peaks],
                linestyle="none", marker="o", ms=5.5, color="#b3261e",
                markeredgecolor="white", markeredgewidth=0.6, zorder=3, label="Peak")
    ax.axhline(0.0, color="#d2dbe6", lw=0.8, zorder=1)
    ax.plot(times[::step], fit.filtered[:stop][::step], color="#16324f", lw=0.9, zorder=2)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(fit.unit or "µV")
    ax.set_title(
        f"{fit.channel} filtered {fit.band[0]:g}–{fit.band[1]:g} Hz"
        f"    ·    {fit.n_cycles} cycles, {fit.n_burst_cycles} burst cycles "
        f"in {fit.n_burst_episodes} episodes"
        + (f"    ·    {fit.burst_duration_s:.2f} s" if fit.n_burst_episodes else ""),
        fontsize=9, loc="left",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    remember_figure(fig, fit.channel, f"{fit.band[0]:g}–{fit.band[1]:g} Hz", fit.unit or "µV")
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
