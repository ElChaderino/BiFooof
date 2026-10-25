"""Figures for the Time tab. The calculations live in analysis."""

from __future__ import annotations

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..analysis.coherence import CoherenceResult, zero_lag_mask
from ..analysis.cycles import CycleFit
from ..analysis.pac import PacResult
from ..analysis.quality import QualitySpan
from ..analysis.shape import BurstStats, aligned_cycles, cycle_cloud
from ..analysis.timefreq import ChannelSeries, PeakTrack, Spectrogram
from ..analysis.settings import format_seconds, reference_phrase
from ..plots import _power_label, clear_figure, mark_spectrum, remember_figure, stamp_figure, style_axes

_TRACE = "#16324f"
_TEAL = "#0e8f86"
_MARK = "#b3261e"
_MUTED = "#63748a"
_ORANGE = "#c2410c"
_BLUE = "#1c4e78"


def _open(fig) -> None:
    clear_figure(fig)
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        return


def draw_spectrogram(
    fig, spec: Spectrogram, episodes, inspection, raw,
    tracks: list[PeakTrack] | None = None,
    color_limits: tuple[float, float] | None = None,
    window=None,
    trace_unit: str = "µV",
) -> None:
    _open(fig)
    if window is None:
        grid = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.15], hspace=0.28)
    else:
        grid = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.95, 1.2], hspace=0.32)
    ax = fig.add_subplot(grid[0])
    finite = spec.power_db[np.isfinite(spec.power_db)]
    if finite.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No spectrogram bins in this window.", ha="center", va="center", color=_MUTED)
    else:
        if color_limits is None:
            low, high = np.percentile(finite, [2, 98])
        else:
            low, high = color_limits
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite) + 1)
        image = ax.pcolormesh(
            spec.times_s, spec.freqs_hz, spec.power_db,
            shading="nearest", cmap="magma", vmin=low, vmax=high,
        )
        bar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.035)
        bar.set_label(f"dB re 1 {spec.power_unit}", fontsize=8, color=_TRACE)
        bar.ax.tick_params(labelsize=7, colors=_MUTED)
        bar.outline.set_edgecolor("#d2dbe6")
        drew_burst = False
        for episode in episodes:
            freq = _episode_hz(episode)
            if freq is None:
                continue
            ax.plot([episode.start_s, episode.end_s], [freq, freq], color="#fffaf3", lw=1.7, solid_capstyle="butt", zorder=4)
            drew_burst = True
        drew_track = False
        if tracks:
            for track in tracks:
                if len(track.times) < 1:
                    continue
                ax.scatter(
                    track.times, track.centers, s=_track_sizes(track.heights),
                    facecolors="none", edgecolors="#fffaf3", linewidths=0.8, zorder=5,
                )
                drew_track = True
        handles = []
        if drew_burst:
            handles.append(Line2D([0], [0], color="#fffaf3", lw=1.7, label="Burst"))
        if drew_track:
            handles.append(Line2D(
                [0], [0], linestyle="none", marker="o", markerfacecolor="none",
                markeredgecolor="#fffaf3", label="Peak track",
            ))
        if handles:
            legend = ax.legend(handles=handles, frameon=True, fontsize=8, loc="upper right")
            legend.get_frame().set_facecolor("#16232f")
            legend.get_frame().set_edgecolor("#16232f")
            for text in legend.get_texts():
                text.set_color("#ffffff")
        _cursor(ax, inspection)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(spec.t_start, max(spec.t_end, spec.times_s[-1] if spec.times_s.size else spec.t_start + 1))
        ax.set_ylim(float(spec.freqs_hz[0]), float(spec.freqs_hz[-1]))
        ax.set_title(
            f"{spec.channel}   ·   {spec.window_s:g} s window, {spec.overlap:.0%} overlap"
            f"   ·   {spec.df_hz:.2f} Hz bins, {spec.dt_s:.2f} s step",
            fontsize=9, loc="left",
        )
        style_axes(ax)
    raw_ax = fig.add_subplot(grid[1])
    _raw(raw_ax, raw, inspection, trace_unit)
    remember_figure(
        fig, spec.channel,
        f"{format_seconds(spec.t_start)}–{format_seconds(spec.t_end)} s",
        spec.power_unit,
    )
    if window is not None:
        _window_spectrum(fig.add_subplot(grid[2]), window)
    stamp_figure(fig)


def draw_trajectory(fig, series: ChannelSeries, tracks: list[PeakTrack], note: str, window=None, inspection=None, top=None) -> None:
    _open(fig)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.2])
    axes = [fig.add_subplot(grid[0])]
    axes.append(fig.add_subplot(grid[1], sharex=axes[0]))
    axes.append(fig.add_subplot(grid[2], sharex=axes[0]))
    _line_series(axes, series, tracks, note, top)
    for ax in axes:
        _cursor(ax, inspection)
    _window_spectrum(fig.add_subplot(grid[3]), window)
    stamp_figure(fig)


def draw_trajectory_with_maps(
    fig, series, tracks, note, names, medians, spreads, window=None, inspection=None,
    top=None, which: str = "median", limits: tuple[float, float] | None = None,
) -> None:
    """Series on the left, one head map on the right, and the clicked window below."""
    _open(fig)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.15], width_ratios=[1.45, 1.0], hspace=0.28, wspace=0.08)
    axes = []
    for row in range(3):
        axes.append(fig.add_subplot(grid[row, 0], sharex=axes[0] if axes else None))
    _line_series(axes, series, tracks, note, top)
    for ax in axes:
        _cursor(ax, inspection)
    from .headmap import draw_head
    spread = which == "spread"
    draw_head(
        fig.add_subplot(grid[0:3, 1]), names,
        spreads if spread else medians,
        "Spread (MAD)" if spread else "Median exponent",
        "plasma" if spread else "viridis",
        series.channel, limits=limits,
    )
    _window_spectrum(fig.add_subplot(grid[3, :]), window)
    stamp_figure(fig)


def draw_raster(fig, fits: list[CycleFit], inspection) -> None:
    _open(fig)
    ax = fig.add_subplot(111)
    if not fits:
        _blank(ax, "No burst episodes in this selection.")
        stamp_figure(fig)
        return
    yticks = []
    labels = []
    placed = []
    for row, fit in enumerate(fits):
        y = len(fits) - 1 - row
        yticks.append(y)
        labels.append(fit.channel)
        ax.axhspan(y - 0.5, y + 0.5, color="#f4f7fb" if row % 2 == 0 else "#ffffff", zorder=0)
        for episode in fit.episodes:
            placed.append((y, episode))
    amps = np.array([
        episode.volt_amp if episode.volt_amp is not None else np.nan
        for _y, episode in placed
    ], dtype=np.float64)
    finite = amps[np.isfinite(amps)]
    mapper = None
    if finite.size:
        low, high = np.percentile(finite, [5, 95])
        if high <= low:
            high = low + 1.0
        mapper = ScalarMappable(norm=Normalize(vmin=float(low), vmax=float(high)), cmap="viridis")
        mapper.set_array([])
    for y, episode in placed:
        width = max(episode.end_s - episode.start_s, 1e-3)
        if mapper is not None and episode.volt_amp is not None and np.isfinite(episode.volt_amp):
            color = mapper.to_rgba(episode.volt_amp)
        else:
            color = _TEAL
        ax.barh(y, width, left=episode.start_s, height=0.62, color=color, align="center", zorder=2)
    if mapper is not None:
        bar = fig.colorbar(mapper, ax=ax, pad=0.01, fraction=0.03)
        unit = fits[0].unit if fits else "µV"
        bar.set_label(f"Amplitude ({unit})", fontsize=8, color=_TRACE)
        bar.ax.tick_params(labelsize=7, colors=_MUTED)
        bar.outline.set_edgecolor("#d2dbe6")
        missing = any(
            episode.volt_amp is None or not np.isfinite(episode.volt_amp)
            for _y, episode in placed
        )
        if missing:
            ax.legend(handles=[Patch(facecolor=_TEAL, edgecolor=_TEAL, label="No amplitude")], frameon=False, fontsize=8)
    starts = [episode.start_s for _y, episode in placed]
    ends = [episode.end_s for _y, episode in placed]
    if starts:
        span = max(ends) - min(starts)
        pad = max(0.4, span * 0.04)
        ax.set_xlim(min(starts) - pad, max(ends) + pad)
    else:
        limits = _fit_limits(fits)
        if limits is not None:
            ax.set_xlim(*limits)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.7, max(len(fits) - 0.3, 0.7))
    _cursor(ax, inspection)
    count = len(placed)
    ax.set_title(
        f"{count} burst episodes   ·   color is amplitude   ·   shared timing is not connectivity",
        fontsize=9, loc="left",
    )
    style_axes(ax, grid=True)
    stamp_figure(fig)


def draw_durations(fig, groups: list[tuple[str, np.ndarray]]) -> None:
    """Histogram and cumulative curve. ``groups`` share one set of bin edges."""
    _open(fig)
    present = [(label, np.asarray(values, dtype=np.float64)) for label, values in groups if np.asarray(values).size]
    if not present:
        _blank(fig.add_subplot(111), "No burst episodes to measure.")
        stamp_figure(fig)
        return
    pooled = np.concatenate([values for _label, values in present])
    count = min(12, max(4, int(np.sqrt(pooled.size))))
    low, high = float(np.min(pooled)), float(np.max(pooled))
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, count + 1)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
    hist = fig.add_subplot(grid[0])
    curve = fig.add_subplot(grid[1], sharex=hist)
    colors = (_BLUE, _ORANGE, _TEAL)
    for index, (label, values) in enumerate(present):
        color = colors[index % len(colors)]
        hist.hist(
            values, bins=edges, color=color, alpha=0.72, edgecolor="white", lw=0.6, label=label,
        )
        ordered = np.sort(values[np.isfinite(values)])
        if ordered.size:
            median = float(np.median(ordered))
            hist.axvline(median, color=color, lw=1.1, ls=":", zorder=3)
            curve.plot(
                ordered, np.arange(1, ordered.size + 1) / ordered.size,
                color=color, lw=1.8, label=label,
            )
            curve.scatter([median], [0.5], s=28, color=color, zorder=4)
    hist.set_ylabel("Episodes")
    hist.legend(frameon=False, fontsize=8)
    curve.set_ylabel("Share of episodes")
    curve.set_xlabel("Burst duration (s)")
    curve.set_ylim(-0.02, 1.05)
    curve.axhline(0.5, color="#d2dbe6", lw=0.8, zorder=1)
    title = present[0][0] if len(present) == 1 else "Same bins, both recordings"
    median = float(np.median(pooled[np.isfinite(pooled)]))
    hist.set_title(
        f"{title}   ·   {pooled.size} episodes   ·   median {median:.2f} s",
        fontsize=9, loc="left",
    )
    for ax in (hist, curve):
        style_axes(ax, grid=True)
    stamp_figure(fig)


def draw_shape(fig, fit: CycleFit, align: str, bursts_only: bool, normalized: bool) -> str:
    _open(fig)
    layout = fig.add_gridspec(2, 4, height_ratios=[2.3, 1.0])
    ax = fig.add_subplot(layout[0, :])
    grid, matrix, hidden = aligned_cycles(fit, align=align, bursts_only=bursts_only, normalized=normalized)
    if matrix.size == 0 or not np.isfinite(matrix).any():
        _blank(ax, "No cycles to overlay.")
    else:
        median = _finite_median(matrix, axis=0)
        finite_rows = np.isfinite(matrix).any(axis=1)
        for row in matrix[finite_rows]:
            ax.plot(grid, row, color=_BLUE, lw=0.45, alpha=0.08, zorder=1)
        low = _finite_percentile(matrix, 10)
        high = _finite_percentile(matrix, 90)
        ax.fill_between(grid, low, high, color=_TEAL, alpha=0.35, lw=0, zorder=2, label="10–90%")
        ax.plot(grid, median, color=_TRACE, lw=2.2, zorder=3, label="Median")
        ax.axhline(0.0, color="#d2dbe6", lw=0.8, zorder=1)
        ax.legend(frameon=False, fontsize=8)
        if normalized:
            ax.set_xlabel("Normalized cycle")
            title = "Resampled trough to trough"
        else:
            ax.set_xlabel("Seconds from the peak" if align == "peak" else "Seconds from the trough")
            title = "Peak aligned" if align == "peak" else "Trough aligned"
        ax.set_ylabel(fit.unit or "µV")
        extra = f"   ·   {hidden} cycles not drawn" if hidden else ""
        ax.set_title(f"{fit.channel}   ·   {title}   ·   {matrix.shape[0]} cycles{extra}", fontsize=9, loc="left")
        style_axes(ax)
    _shape_histograms(fig, layout, fit, bursts_only)
    stamp_figure(fig)
    return ""


def draw_scatter(fig, groups: list[tuple[str, CycleFit, str]], bursts_only: bool) -> None:
    """One cloud per recording. A line is fit to the points that are drawn."""
    _open(fig)
    ax = fig.add_subplot(111)
    drawn_x = []
    drawn_y = []
    any_points = False
    for label, fit, color in groups:
        freqs, amps, bursts = cycle_cloud(fit)
        if bursts_only:
            freqs, amps, bursts = freqs[bursts], amps[bursts], bursts[bursts]
        if freqs.size == 0:
            continue
        any_points = True
        if len(groups) == 1:
            other = ~bursts
            if np.any(other):
                ax.scatter(
                    freqs[other], amps[other], s=28, color="#9aafc4", alpha=0.75,
                    linewidths=0.4, edgecolors="white", label="Cycle", zorder=2,
                )
            if np.any(bursts):
                ax.scatter(
                    freqs[bursts], amps[bursts], s=36, color=_TEAL, alpha=0.9,
                    linewidths=0.4, edgecolors="white", label="Burst cycle", zorder=3,
                )
        else:
            ax.scatter(
                freqs, amps, s=32, color=color, alpha=0.8,
                linewidths=0.4, edgecolors="white", label=label, zorder=3,
            )
        drawn_x.append(freqs)
        drawn_y.append(amps)
    if not any_points:
        _blank(ax, "No cycles to plot.")
        stamp_figure(fig)
        return
    cloud_x = np.concatenate(drawn_x)
    cloud_y = np.concatenate(drawn_y)
    _describe_cloud(ax, cloud_x, cloud_y)
    _frame_cloud(ax, cloud_x, cloud_y)
    ax.set_xlabel("Cycle frequency (Hz), axis is the central 96%")
    unit = groups[0][1].unit if groups else "µV"
    ax.set_ylabel(f"Amplitude ({unit}), axis is the central 96%")
    title = groups[0][1].channel if len(groups) == 1 else "Colored by recording"
    count = int(np.isfinite(cloud_x).sum())
    ax.set_title(
        f"{title}   ·   {count} cycles   ·   frame is the central spread",
        fontsize=9, loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, grid=True)
    stamp_figure(fig)


def draw_reactivity(fig, payload: dict, inspection) -> None:
    _open(fig)
    kind = payload.get("kind")
    if kind == "events":
        extras = list(payload.get("extras") or [])
        axes = fig.subplots(1 + len(extras), 1, sharex=True, squeeze=False).ravel()
        _event_course(axes[0], payload["grid"], payload["curves"], _power_axis(payload), payload.get("title", ""))
        for ax, extra in zip(axes[1:], extras):
            _event_course(ax, extra["grid"], extra["curves"], extra["ylabel"], "")
        axes[-1].set_xlabel("Seconds from the mark")
    elif kind == "separate":
        axes = fig.subplots(1, 2, sharey=True)
        for ax, key, color, label in (
            (axes[0], "left", _BLUE, payload.get("left_label", "Left")),
            (axes[1], "right", _ORANGE, payload.get("right_label", "Right")),
        ):
            times, power = payload[key]
            ax.plot(times, power, color=color, lw=1.3)
            ax.set_title(label, fontsize=9, loc="left")
            ax.set_xlabel("Time (s)")
            style_axes(ax, grid=True)
        axes[0].set_ylabel(_power_axis(payload))
    else:
        ax = fig.add_subplot(111)
        times, power = payload.get("single", (np.zeros(0), np.zeros(0)))
        if times.size == 0:
            _blank(ax, payload.get("note") or "No band power in this window.")
        else:
            ax.plot(times, power, color=_BLUE, lw=1.3)
            _cursor(ax, inspection)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(_power_axis(payload))
            ax.set_title(payload.get("title", ""), fontsize=9, loc="left")
            style_axes(ax, grid=True)
    stamp_figure(fig)


def draw_coherence(fig, result: CoherenceResult, raw=None, other: CoherenceResult | None = None, other_label: str = "", trace_unit: str = "µV") -> None:
    _open(fig)
    rows = 4 if raw is not None else 3
    axes = fig.subplots(rows, 1, sharex=False)
    this_label = "This file" if other is not None else "Coherence"
    other_name = other_label or "Other file"
    _coherence_bands(axes[0], result)
    axes[0].plot(result.freqs_hz, result.coherence, color=_BLUE, lw=1.4, label=this_label)
    if result.wpli is not None and other is None:
        axes[0].plot(result.freqs_hz, result.wpli, color=_TEAL, lw=1.1, ls="--", label="wPLI")
    if other is not None:
        axes[0].plot(other.freqs_hz, other.coherence, color=_ORANGE, lw=1.3, label=other_name)
    lag = zero_lag_mask(result)
    if np.any(lag):
        axes[0].plot(result.freqs_hz[lag], result.coherence[lag], linestyle="none", marker="o",
                     ms=3.5, color=_ORANGE if other is None else _TEAL, label="Zero lag")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Coherence")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].plot(result.freqs_hz, result.phase_deg, color=_BLUE, lw=1.1, label=this_label)
    if other is not None:
        axes[1].plot(other.freqs_hz, other.phase_deg, color=_ORANGE, lw=1.1, label=other_name)
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(-180, 180)
    axes[1].set_ylabel("Phase (deg), wrapped")
    axes[2].plot(result.freqs_hz, result.imaginary, color=_BLUE if other is not None else _ORANGE, lw=1.2, label=this_label if other is not None else None)
    if other is not None:
        axes[2].plot(other.freqs_hz, other.imaginary, color=_ORANGE, lw=1.1, label=other_name)
        axes[2].legend(frameon=False, fontsize=8)
    axes[2].axhline(0.0, color="#d2dbe6", lw=0.8)
    axes[2].set_ylabel("Imag. coh.")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[0].set_title(
        f"{result.channel_a} × {result.channel_b}   ·   {reference_phrase(result.reference)}",
        fontsize=9, loc="left",
    )
    for ax in axes[:3]:
        style_axes(ax, grid=True)
    if raw is not None:
        _pair_raw(axes[3], raw, trace_unit)
    remember_figure(fig, f"{result.channel_a} × {result.channel_b}", reference_phrase(result.reference))
    stamp_figure(fig)


def draw_pac(fig, result: PacResult) -> None:
    _open(fig)
    if result.histogram is None and result.comodulogram is None:
        ax = fig.add_subplot(111)
        _blank(ax, result.note or "No coupling estimate.")
        stamp_figure(fig)
        return
    if result.comodulogram is not None and result.phase_centers is not None and result.amp_centers is not None:
        axes = fig.subplots(1, 2, width_ratios=[1.0, 1.35])
        _pac_hist(axes[0], result)
        image = axes[1].pcolormesh(
            result.phase_centers, result.amp_centers, result.comodulogram,
            shading="nearest", cmap="magma",
        )
        bar = fig.colorbar(image, ax=axes[1], pad=0.02, fraction=0.046)
        bar.set_label("MI", fontsize=8)
        bar.ax.tick_params(labelsize=7, colors=_MUTED)
        bar.outline.set_edgecolor("#d2dbe6")
        axes[1].axvline(result.phase_band[0], color=_BLUE, lw=1.0, ls="--", label="Phase band")
        axes[1].axvline(result.phase_band[1], color=_BLUE, lw=1.0, ls="--")
        axes[1].axhline(result.amp_band[0], color=_TEAL, lw=1.0, ls="--", label="Amplitude band")
        axes[1].axhline(result.amp_band[1], color=_TEAL, lw=1.0, ls="--")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].set_xlabel("Phase frequency (Hz)")
        axes[1].set_ylabel("Amplitude frequency (Hz)")
        who = ""
        if result.phase_channel and result.amplitude_channel:
            who = f"{result.phase_channel} phase, {result.amplitude_channel} amplitude   ·   "
        axes[1].set_title(f"{who}uncorrected   ·   {_cell_count(result)}", fontsize=9, loc="left")
        style_axes(axes[1])
    else:
        ax = fig.add_subplot(111)
        _pac_hist(ax, result)
    stamp_figure(fig)


def draw_quality(fig, recording_names: list[str], spans: list[QualitySpan], inspection) -> None:
    _open(fig)
    ax = fig.add_subplot(111)
    if not recording_names:
        _blank(ax, "No channels.")
        stamp_figure(fig)
        return
    kinds = (
        ("amplitude", _MARK, ""),
        ("step", _MARK, "//"),
        ("flat", _MARK, ".."),
        ("clip", _MARK, "xx"),
        ("muscle", _ORANGE, ""),
        ("drift", _ORANGE, "//"),
        ("mains", _ORANGE, "xx"),
        ("file mark", _BLUE, ""),
        ("missing", _BLUE, "//"),
        ("left out", _TEAL, ""),
        ("user", _TEAL, "//"),
    )
    style_of = {kind: (color, hatch) for kind, color, hatch in kinds}
    lane_of = {kind: index for index, (kind, _color, _hatch) in enumerate(kinds)}
    lane_height = 0.78 / len(kinds)
    y_of = {name: len(recording_names) - 1 - index for index, name in enumerate(recording_names)}
    for span in spans:
        if span.channel and span.channel not in y_of:
            continue
        if span.channel:
            rows = [y_of[span.channel]]
        else:
            rows = list(y_of.values())
        color, hatch = style_of.get(span.kind, (_MUTED, ""))
        lane = lane_of.get(span.kind, 0)
        for y in rows:
            center = y - 0.39 + (lane + 0.5) * lane_height
            ax.barh(
                center, max(span.end_s - span.start_s, 1e-3), left=span.start_s,
                height=lane_height * 0.86, color=color, hatch=hatch,
                edgecolor="#ffffff", linewidth=0.4, align="center", zorder=2,
            )
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels(list(y_of.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.8, max(len(recording_names) - 0.2, 0.8))
    _cursor(ax, inspection)
    shown = []
    for kind, color, hatch in kinds:
        if not any(span.kind == kind for span in spans):
            continue
        shown.append(Patch(
            facecolor=color, edgecolor="#16232f", hatch=hatch, label=kind,
        ))
    if shown:
        ax.legend(handles=shown, frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax.set_title("Each kind has its own stripe.", fontsize=9, loc="left")
    style_axes(ax)
    stamp_figure(fig)


def _line_series(axes, series: ChannelSeries, tracks: list[PeakTrack], note: str, top=None) -> None:
    if top is None:
        values, label = series.exponent, "Exponent"
    else:
        values, label = top
    _series(axes[0], series.times_s, values, series.weak, label, _BLUE, "Weak fit")
    _series(axes[1], series.times_s, series.peak_height, series.weak | series.at_floor, "Peak height (log10)", _ORANGE, "Weak fit")
    _series(axes[2], series.times_s, series.peak_hz, series.weak | series.at_floor, "Peak frequency (Hz)", _TRACE, "Weak fit")
    for track in tracks:
        if len(track.centers) >= 2:
            axes[2].plot(track.times, track.centers, color=_BLUE, lw=1.1, zorder=2)
        if track.centers:
            axes[2].scatter(
                track.times, track.centers, s=_track_sizes(track.heights),
                color=_BLUE, linewidths=0, zorder=3,
            )
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"{series.channel}   ·   {note or 'each point is one fit window'}", fontsize=9, loc="left")
    for ax in axes:
        style_axes(ax, grid=True)


def _series(ax, times, values, weak, label, color, marker_label: str | None = None) -> None:
    ax.plot(times, values, color=color, lw=1.3, zorder=2)
    if weak is not None and np.any(weak & np.isfinite(values)):
        ax.plot(times[weak], values[weak], linestyle="none", marker="o", ms=4,
                markerfacecolor="white", markeredgecolor=color, zorder=3, label=marker_label)
        if marker_label:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylabel(label, fontsize=8)


def _power_axis(payload: dict) -> str:
    return f"Band power ({payload.get('unit') or 'µV²/Hz'})"


def _coherence_bands(ax, result: CoherenceResult) -> None:
    if result.freqs_hz.size == 0:
        return
    low = float(result.freqs_hz[0])
    high = float(result.freqs_hz[-1])
    for start, stop, _name, color in (
        (1.0, 4.0, "delta", "#d9e4f2"),
        (4.0, 8.0, "theta", "#d7efe9"),
        (8.0, 12.0, "alpha", "#c9dcf0"),
        (13.0, 30.0, "beta", "#f3e4d4"),
        (30.0, 45.0, "gamma", "#e6dff0"),
    ):
        left, right = max(start, low), min(stop, high)
        if right > left:
            ax.axvspan(left, right, color=color, zorder=0, lw=0)


def _raw(ax, raw, inspection, unit: str = "µV") -> None:
    if not raw or raw[0].size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Click the spectrogram to see that stretch.", ha="center", va="center", color=_MUTED, fontsize=9)
        return
    times, values = raw[0], raw[1]
    ax.plot(times, values, color=_TRACE, lw=0.8)
    if len(raw) > 2 and raw[2] is not None:
        rejected = ~np.asarray(raw[2], dtype=bool)
        if np.any(rejected):
            ax.plot(
                times[rejected], values[rejected], linestyle="none", marker=".",
                ms=2.4, color=_MARK, zorder=3,
            )
    if inspection is not None:
        center, span = inspection
        ax.axvspan(center - span / 2.0, center + span / 2.0, color=_TEAL, alpha=0.15, lw=0)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel(unit or "µV")
    ax.set_xlabel("Time (s)")
    style_axes(ax)


def _window_spectrum(ax, window) -> None:
    """The specparam fit of the sliding window under the cursor."""
    if window is None:
        _blank(ax, "Click a time to see that window's spectrum.")
        return
    fit, item, message = window
    if message:
        _blank(ax, message)
        return
    if fit is None or item is None or item.freqs is None or item.spectrum is None:
        _blank(ax, (item.reason if item is not None and item.reason else "No fit for this window."))
        return
    mark_spectrum(ax, fit.fmin, fit.fmax)
    ax.plot(item.freqs, item.spectrum, color=_TRACE, lw=1.5, label="Spectrum", zorder=2)
    if item.model is not None:
        ax.plot(item.freqs, item.model, color=_ORANGE, lw=1.6, label="Model", zorder=3)
    if item.aperiodic is not None:
        ax.plot(item.freqs, item.aperiodic, color=_TEAL, lw=1.2, ls="--", label="Aperiodic", zorder=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(_power_label(fit))
    exponent = "" if item.exponent is None else f"   ·   exponent {item.exponent:.2f}"
    ax.set_title(
        f"{item.name}   ·   {format_seconds(fit.t_start)}–{format_seconds(fit.t_end)} s{exponent}",
        fontsize=9, loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax, grid=True)


def _cell_count(result: PacResult) -> str:
    if result.comodulogram is None or result.n_cells <= 0:
        return "uncorrected"
    finite = int(np.isfinite(result.comodulogram).sum())
    if finite == result.n_cells:
        return f"{result.n_cells} cells, uncorrected"
    return f"{finite} of {result.n_cells} cells, uncorrected"


def _cursor(ax, inspection) -> None:
    if inspection is None:
        return
    ax.axvline(inspection[0], color="#16232f", lw=0.7, ls="--", alpha=0.7, zorder=6)


def _pac_hist(ax, result: PacResult) -> None:
    if result.histogram is None or result.bin_edges is None:
        _blank(ax, result.note or "No histogram.")
        return
    centers = 0.5 * (result.bin_edges[:-1] + result.bin_edges[1:])
    ax.bar(centers, result.histogram, width=np.diff(result.bin_edges) * 0.9, color="#d6e4f0", edgecolor=_BLUE, lw=0.6)
    mi = "—" if result.mi is None else f"{result.mi:.3f}"
    p_text = "—" if result.p_value is None else f"{result.p_value:.3f}"
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Mean amplitude")
    ax.set_title(
        f"MI {mi}   ·   surrogate p {p_text}   ·   {result.phase_band[0]:g}–{result.phase_band[1]:g} Hz phase, "
        f"{result.amp_band[0]:g}–{result.amp_band[1]:g} Hz amplitude",
        fontsize=9, loc="left",
    )
    style_axes(ax)


def _event_course(ax, grid, curves, ylabel: str, title: str) -> None:
    if grid is None or curves is None or np.size(curves) == 0:
        _blank(ax, "No values in this window.")
        return
    for row in curves:
        ax.plot(grid, row, color="#9aafc4", lw=0.8, alpha=0.8)
    ax.plot(grid, _finite_median(curves, axis=0), color=_BLUE, lw=1.8, label="Median")
    ax.axvline(0.0, color=_MARK, lw=0.8, ls="--")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
    style_axes(ax, grid=True)


def _pair_raw(ax, raw, unit: str = "µV") -> None:
    ax.set_label("raw")
    duration = None
    stretches = []
    for stretch in raw or []:
        if stretch and stretch[0] == "empty":
            duration = float(stretch[1])
            continue
        stretches.append(stretch)
    colors = (_BLUE, _ORANGE)
    plotted = False
    for index, stretch in enumerate(stretches):
        times, values = stretch[0], stretch[1]
        if getattr(times, "size", 0) == 0:
            continue
        plotted = True
        ax.plot(times, values, color=colors[index % 2], lw=0.8, label=stretch[3] if len(stretch) > 3 else "")
        if len(stretch) > 2 and stretch[2] is not None:
            rejected = ~np.asarray(stretch[2], dtype=bool)
            if np.any(rejected):
                ax.plot(times[rejected], values[rejected], linestyle="none", marker=".", ms=2.2, color=_MARK, zorder=3)
    if not plotted:
        ax.set_xlim(0, duration or 1)
        ax.text(
            0.5, 0.5, "Click this row to choose a time.",
            transform=ax.transAxes, ha="center", va="center", color=_MUTED, fontsize=9,
        )
    ax.set_ylabel(unit or "µV")
    ax.set_xlabel("Time (s)")
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    style_axes(ax)


def _shape_histograms(fig, layout, fit: CycleFit, bursts_only: bool) -> None:
    fields = (
        ("volt_amp", f"Amplitude ({fit.unit or 'µV'})"),
        ("period_ms", "Period (ms)"),
        ("rise_decay", "Rise/decay"),
        ("peak_trough", "Peak/trough"),
    )
    rows = [
        row for row in fit.cycles
        if not bursts_only or row.get("is_burst")
    ]
    for column, (key, label) in enumerate(fields):
        ax = fig.add_subplot(layout[1, column])
        values = np.asarray([row.get(key) for row in rows if row.get(key) is not None], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.axis("off")
            ax.set_title(label, fontsize=8, loc="left")
            continue
        ax.hist(
            values, bins=min(12, max(4, int(np.sqrt(values.size)))),
            color="#d6e4f0", edgecolor=_BLUE, lw=0.6,
        )
        ax.axvline(float(np.median(values)), color=_TRACE, lw=1.0, ls=":")
        ax.set_title(label, fontsize=8, loc="left")
        style_axes(ax)


def _describe_cloud(ax, freqs, amps) -> None:
    finite = np.isfinite(freqs) & np.isfinite(amps)
    freqs = np.asarray(freqs, dtype=np.float64)[finite]
    amps = np.asarray(amps, dtype=np.float64)[finite]
    if freqs.size < 3 or float(np.max(freqs) - np.min(freqs)) < 1e-3:
        return
    slope, intercept = np.polyfit(freqs, amps, 1)
    order = np.argsort(freqs)
    line = slope * freqs[order] + intercept
    residual = amps[order] - line
    low, high = np.percentile(residual, [10, 90])
    ax.fill_between(freqs[order], line + low, line + high, color="#d5dde6", alpha=0.7, lw=0, zorder=1)
    ax.plot(freqs[order], line, color=_TRACE, lw=1.2, zorder=4, label="Line through every point")


def _frame_cloud(ax, freqs, amps) -> None:
    """Keep one extreme cycle from flattening the cloud. The line still uses every point."""
    finite = np.isfinite(freqs) & np.isfinite(amps)
    freqs = np.asarray(freqs, dtype=np.float64)[finite]
    amps = np.asarray(amps, dtype=np.float64)[finite]
    if freqs.size < 5:
        return
    x_low, x_high = np.percentile(freqs, [2, 98])
    y_low, y_high = np.percentile(amps, [2, 98])
    x_pad = max((x_high - x_low) * 0.08, 0.15)
    y_pad = max((y_high - y_low) * 0.08, 0.5)
    floor = float(y_low - y_pad)
    if y_low >= 0:
        floor = max(0.0, floor)
    ax.set_xlim(float(x_low - x_pad), float(x_high + x_pad))
    ax.set_ylim(floor, float(y_high + y_pad))


def _track_sizes(heights) -> np.ndarray:
    values = np.asarray(heights, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(values.shape, 16.0)
    low, high = float(np.min(finite)), float(np.max(finite))
    if high - low < 1e-9:
        scaled = np.ones(values.shape)
    else:
        scaled = (values - low) / (high - low)
        scaled = np.where(np.isfinite(scaled), scaled, 0.35)
    return 8.0 + 34.0 * scaled


def _episode_hz(episode) -> float | None:
    if episode.period_ms is None or episode.period_ms <= 0:
        return None
    return 1000.0 / float(episode.period_ms)


def _finite_median(values, axis: int = 0) -> np.ndarray:
    """Median along ``axis``. An all-empty column stays NaN and does not warn."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        finite = data[np.isfinite(data)]
        return np.asarray(np.nan if finite.size == 0 else np.median(finite))
    if axis != 0:
        data = np.moveaxis(data, axis, 0)
    out = np.full(data.shape[1:], np.nan, dtype=np.float64)
    for index in np.ndindex(out.shape):
        column = data[(slice(None),) + index]
        finite = column[np.isfinite(column)]
        if finite.size:
            out[index] = float(np.median(finite))
    return out


def _finite_percentile(values, percent: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    out = np.full(data.shape[1], np.nan, dtype=np.float64)
    for index in range(data.shape[1]):
        finite = data[:, index]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            out[index] = float(np.percentile(finite, percent))
    return out


def _fit_limits(fits: list[CycleFit]) -> tuple[float, float] | None:
    """Recording-clock span of the fits, used when a burst row has no episodes."""
    bounds = []
    for fit in fits:
        if fit.sfreq <= 0 or fit.filtered.size == 0:
            continue
        start = float(fit.t_offset)
        end = start + fit.filtered.size / float(fit.sfreq)
        if end > start:
            bounds.append((start, end))
    if not bounds:
        return None
    return min(item[0] for item in bounds), max(item[1] for item in bounds)


def _blank(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=_MUTED, fontsize=11)


def duration_caption(fit: CycleFit, stats: BurstStats) -> str:
    occupancy = "—" if stats.occupancy is None else f"{100.0 * stats.occupancy:.1f}% of accepted time"
    median = "—" if stats.median_s is None else f"{stats.median_s:.2f} s"
    iqr = "—"
    if stats.q1_s is not None and stats.q3_s is not None:
        iqr = f"{stats.q1_s:.2f}–{stats.q3_s:.2f} s"
    per_minute = "—" if stats.per_minute is None else f"{stats.per_minute:.1f}"
    cycles = "—" if stats.median_cycles is None else f"{stats.median_cycles:.0f}"
    gap = "—" if stats.gaps_s.size == 0 else f"{float(np.median(stats.gaps_s)):.2f} s"
    return (
        f"{fit.channel}  ·  {stats.n_episodes} episodes  ·  median {median}  ·  "
        f"IQR {iqr}  ·  {per_minute} per minute  ·  {occupancy}  ·  "
        f"median {cycles} cycles  ·  median gap {gap}"
    )
