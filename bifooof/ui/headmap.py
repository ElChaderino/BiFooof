"""Small 10-20 head maps for spectral parameters."""

from __future__ import annotations

import numpy as np

# Unit-circle positions, nose at the top. Legacy temporal names share the modern site.
HEAD: dict[str, tuple[float, float]] = {
    "Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95),
    "F7": (-0.81, 0.59), "F3": (-0.40, 0.55), "Fz": (0.0, 0.59),
    "F4": (0.40, 0.55), "F8": (0.81, 0.59),
    "T7": (-1.0, 0.0), "C3": (-0.50, 0.0), "Cz": (0.0, 0.0), "C4": (0.50, 0.0), "T8": (1.0, 0.0),
    "P7": (-0.81, -0.59), "P3": (-0.40, -0.55), "Pz": (0.0, -0.59),
    "P4": (0.40, -0.55), "P8": (0.81, -0.59),
    "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
}
HEAD["T3"] = HEAD["T7"]
HEAD["T4"] = HEAD["T8"]
HEAD["T5"] = HEAD["P7"]
HEAD["T6"] = HEAD["P8"]


def draw_head(ax, names: list[str], values: list[float | None], title: str,
              cmap: str, selected: str | None = None, marked: list[bool] | None = None,
              limits: tuple[float, float] | None = None) -> None:
    """Draw one parameter on a 10-20 head. Channels outside the montage are skipped."""
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="#16232f", lw=1.0)
    ax.plot([-0.12, 0.0, 0.12], [0.98, 1.16, 0.98], color="#16232f", lw=1.0)
    ax.plot([-1.02, -1.14, -1.14, -1.02], [0.18, 0.12, -0.12, -0.18], color="#16232f", lw=1.0)
    ax.plot([1.02, 1.14, 1.14, 1.02], [0.18, 0.12, -0.12, -0.18], color="#16232f", lw=1.0)

    points = []
    for name, value in zip(names, values):
        if name not in HEAD or value is None or not np.isfinite(value):
            if name in HEAD:
                x, y = HEAD[name]
                ax.scatter([x], [y], s=160, color="#d5dde6", zorder=2)
                _ring(ax, name, x, y, selected)
                _mark(ax, name, x, y, names, marked)
                ax.annotate(name, (x, y), ha="center", va="center", fontsize=6, color="#16232f")
            continue
        points.append((name, HEAD[name][0], HEAD[name][1], float(value)))

    if points:
        vals = np.array([item[3] for item in points], dtype=float)
        if limits is not None and np.isfinite(limits[0]) and np.isfinite(limits[1]) and limits[1] > limits[0]:
            low, high = float(limits[0]), float(limits[1])
        else:
            low = float(np.min(vals))
            high = float(np.max(vals))
        if high - low < 1e-9:
            low -= 0.5
            high += 0.5
        mapper = ax.scatter(
            [item[1] for item in points],
            [item[2] for item in points],
            c=vals, s=220, cmap=cmap, vmin=low, vmax=high, zorder=3,
            edgecolors="#ffffff", linewidths=0.6,
        )
        for name, x, y, _value in points:
            _ring(ax, name, x, y, selected)
            _mark(ax, name, x, y, names, marked)
            weight = "bold" if name == selected else "regular"
            ax.annotate(name, (x, y), ha="center", va="center", fontsize=6,
                        color="#16232f", fontweight=weight, zorder=4)
        bar = ax.figure.colorbar(mapper, ax=ax, fraction=0.046, pad=0.04, shrink=0.78)
        bar.ax.tick_params(labelsize=7, length=2, colors="#63748a")
        bar.outline.set_edgecolor("#d2dbe6")
        bar.outline.set_linewidth(0.6)

    ax.set_title(title, fontsize=9, color="#16232f")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.25, 1.4)


def _mark(ax, name: str, x: float, y: float, names: list[str], marked: list[bool] | None) -> None:
    if marked is None or name not in names:
        return
    index = names.index(name)
    if index < len(marked) and marked[index]:
        ax.scatter([x], [y], s=420, facecolors="none", edgecolors="#9a3412",
                   linewidths=1.3, zorder=5)


def _ring(ax, name: str, x: float, y: float, selected: str | None) -> None:
    if name != selected:
        return
    ax.scatter([x], [y], s=360, facecolors="none", edgecolors="#1c4e78",
               linewidths=1.5, zorder=5)
