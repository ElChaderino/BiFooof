"""Matplotlib canvas embedded in the Qt window."""

from __future__ import annotations

import os

import matplotlib

if os.environ.get("BIFOOF_HEADLESS") != "1" and "qtagg" not in matplotlib.get_backend().lower():
    matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 8.0, height: float = 4.8, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
        super().__init__(self.fig)
        self.setMinimumHeight(200)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        engine = self.fig.get_layout_engine()
        if engine is None or engine.__class__.__name__ != "ConstrainedLayoutEngine":
            self.fig.set_layout_engine("tight")
        self.draw_idle()

    def clear(self) -> None:
        clear_figure(self.fig)


from ..plots import clear_figure, finish_figure, style_axes  # noqa: E402  re-export for the panels
