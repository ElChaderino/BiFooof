"""Paged multichannel trace viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QScrollBar,
    QVBoxLayout, QWidget,
)

from ..io.recording import Recording
from ..plots import clear_figure, stamp_figure
from .canvas import MplCanvas, style_axes

_WINDOWS = [("5 s", 5.0), ("10 s", 10.0), ("15 s", 15.0), ("20 s", 20.0), ("30 s", 30.0)]
_GAINS = [("50 µV", 50.0), ("75 µV", 75.0), ("100 µV", 100.0),
          ("150 µV", 150.0), ("200 µV", 200.0), ("300 µV", 300.0)]
_TRACE = "#16324f"
_SELECTED = "#c2410c"


class TracePanel(QWidget):
    """Scroll a page of EEG. Checked channels are drawn; the current row is selected."""

    channel_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.recording: Recording | None = None
        self.condition_label = ""
        self._selected = 0
        self._t0 = 0.0
        self._win_s = 10.0
        self._gain = 100.0
        self._visible: list[int] = []
        self._page: dict[int, float] = {}
        self._hidden: dict[int, set[str]] = {}

        self.channels = QListWidget()
        self.channels.setFixedWidth(168)
        self.channels.currentRowChanged.connect(self._on_row)
        self.channels.itemChanged.connect(lambda _item: self._render())

        self.window = QComboBox()
        for label, value in _WINDOWS:
            self.window.addItem(label, value)
        self.window.setCurrentIndex(1)
        self.window.setToolTip("How many seconds are drawn on one page.")
        self.window.currentIndexChanged.connect(self._on_window)

        self.gain = QComboBox()
        for label, value in _GAINS:
            self.gain.addItem(label, value)
        self.gain.setCurrentIndex(2)
        self.gain.setToolTip("Microvolts from the center of a row to the next row.")
        self.gain.currentIndexChanged.connect(self._on_gain)

        controls = QHBoxLayout()
        window_label = QLabel("Window")
        window_label.setObjectName("Muted")
        gain_label = QLabel("Gain")
        gain_label.setObjectName("Muted")
        controls.addWidget(window_label)
        controls.addWidget(self.window)
        controls.addWidget(gain_label)
        controls.addWidget(self.gain)
        controls.addStretch(1)
        hint = QLabel("Check a channel to show it. Click it to select it for the fits.")
        hint.setObjectName("Muted")
        controls.addWidget(hint)

        self.canvas = MplCanvas(width=9.0, height=5.2)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.setEnabled(False)
        self.scroll.valueChanged.connect(self._on_scroll)

        right = QVBoxLayout()
        right.addLayout(controls)
        right.addWidget(self.canvas, 1)
        right.addWidget(self.scroll)

        side = QVBoxLayout()
        side.setSpacing(6)
        heading = QLabel("CHANNELS")
        heading.setObjectName("H2")
        side.addWidget(heading)
        side.addWidget(self.channels, 1)

        root = QHBoxLayout(self)
        root.addLayout(side)
        root.addLayout(right, 1)
        self._empty("Open a recording to see the traces.")

    def set_recording(self, recording: Recording, selected: int | None = None) -> None:
        self._remember()
        self.recording = recording
        self._t0 = self._page.get(id(recording), 0.0)
        chosen = 0
        if selected is not None and recording.n_channels:
            chosen = max(0, min(int(selected), recording.n_channels - 1))
        self._selected = chosen
        hidden = self._hidden.get(id(recording), set())
        self.channels.blockSignals(True)
        self.channels.clear()
        for name in recording.ch_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if name in hidden else Qt.Checked)
            self.channels.addItem(item)
        if recording.n_channels:
            self.channels.setCurrentRow(chosen)
        self.channels.blockSignals(False)
        self._configure_scroll()
        self._render()

    def set_selected(self, index: int) -> None:
        if self.recording is None or not 0 <= index < self.recording.n_channels:
            return
        if index == self._selected and self.channels.currentRow() == index:
            self._render()
            return
        self._selected = index
        self.channels.blockSignals(True)
        self.channels.setCurrentRow(index)
        self.channels.blockSignals(False)
        self._render()

    def show_time(self, t_center: float) -> None:
        """Scroll so ``t_center`` sits in the page. This does not change the fit window."""
        if self.recording is None:
            return
        duration = self.recording.duration_s
        max_t0 = max(0.0, duration - self._win_s)
        self._t0 = min(max(0.0, float(t_center) - self._win_s / 2.0), max_t0)
        self._configure_scroll()
        self._render()

    def _remember(self) -> None:
        recording = self.recording
        if recording is None:
            return
        key = id(recording)
        self._page[key] = self._t0
        hidden = set()
        for index in range(self.channels.count()):
            item = self.channels.item(index)
            if item is not None and item.checkState() != Qt.Checked:
                hidden.add(item.text())
        self._hidden[key] = hidden

    def _on_row(self, row: int) -> None:
        if row < 0 or self.recording is None:
            return
        self._selected = row
        self._render()
        self.channel_selected.emit(row)

    def _on_window(self, _index: int = 0) -> None:
        self._win_s = float(self.window.currentData())
        self._configure_scroll()
        self._render()

    def _on_gain(self, _index: int = 0) -> None:
        self._gain = float(self.gain.currentData())
        self._render()

    def _configure_scroll(self) -> None:
        duration = 0.0 if self.recording is None else self.recording.duration_s
        max_t0 = max(0.0, duration - self._win_s)
        self._t0 = min(self._t0, max_t0)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, int(round(max_t0 * 10)))
        self.scroll.setPageStep(max(1, int(round(self._win_s * 10))))
        self.scroll.setSingleStep(10)
        self.scroll.setValue(int(round(self._t0 * 10)))
        self.scroll.setEnabled(max_t0 > 0)
        self.scroll.blockSignals(False)

    def _on_scroll(self, value: int) -> None:
        self._t0 = value / 10.0
        self._render()

    def _checked(self) -> list[int]:
        chosen = []
        for index in range(self.channels.count()):
            item = self.channels.item(index)
            if item is not None and item.checkState() == Qt.Checked:
                chosen.append(index)
        return chosen

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.ydata is None or not self._visible:
            return
        offsets = [(len(self._visible) - 1 - row) for row in range(len(self._visible))]
        nearest = int(np.argmin([abs(event.ydata - offset) for offset in offsets]))
        self.channel_selected.emit(self._visible[nearest])

    def _empty(self, message: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#63748a", fontsize=12)
        self.canvas.draw_idle()

    def _render(self) -> None:
        recording = self.recording
        if recording is None:
            return
        visible = self._checked()
        self._visible = visible
        if not visible:
            self._empty("Check at least one channel to draw the page.")
            return

        sfreq = recording.sfreq
        start = int(round(self._t0 * sfreq))
        stop = min(int(round((self._t0 + self._win_s) * sfreq)), recording.n_samples)
        if stop <= start:
            self._empty("This page has no samples.")
            return
        times = np.arange(start, stop) / sfreq
        step = max(1, len(times) // 2500)
        times = times[::step]
        segment = recording.data[visible, start:stop][:, ::step]

        fig = self.canvas.fig
        clear_figure(fig)
        ax = fig.add_subplot(111)
        separation = 1.0
        scale = separation / (2.0 * self._gain)
        count = segment.shape[0]
        offsets = [(count - 1 - row) * separation for row in range(count)]
        for offset in offsets:
            ax.axhline(offset, color="#e4eaf1", lw=0.6, zorder=0)
        for row, channel_index in enumerate(visible):
            color = _SELECTED if channel_index == self._selected else _TRACE
            width = 1.15 if channel_index == self._selected else 0.7
            clipped = np.clip(segment[row] * scale, -separation, separation)
            ax.plot(times, clipped + offsets[row], color=color, lw=width, zorder=2)

        names = [recording.ch_names[index] for index in visible]
        ax.set_yticks(offsets)
        ax.set_yticklabels(names, fontsize=8)
        for label, channel_index in zip(ax.get_yticklabels(), visible):
            if channel_index == self._selected:
                label.set_color(_SELECTED)
                label.set_fontweight("bold")
        ax.set_ylim(-separation * 0.85, (count - 0.15) * separation)
        ax.set_xlim(times[0], times[-1])
        bar_x = times[0] + (times[-1] - times[0]) * 0.012
        bar_y = -separation * 0.55
        ax.plot([bar_x, bar_x], [bar_y, bar_y + 0.5], color="#16232f", lw=1.6, solid_capstyle="butt", zorder=3)
        ax.text(bar_x + (times[-1] - times[0]) * 0.012, bar_y + 0.25, f"{self._gain:.0f} µV",
                va="center", ha="left", fontsize=7, color="#63748a")
        ax.set_xlabel("Time (s)")
        end = min(self._t0 + self._win_s, recording.duration_s)
        prefix = f"{self.condition_label}   ·   " if self.condition_label else ""
        hidden = "" if self._selected in visible else "   ·   selected channel is hidden"
        ax.set_title(
            f"{prefix}{self._t0:.1f}–{end:.1f} s of {recording.duration_s:.0f} s"
            f"   ·   {recording.sfreq:.0f} Hz   ·   {recording.source_format.upper()}{hidden}",
            fontsize=9, loc="left", color="#16232f",
        )
        style_axes(ax)
        fig.tight_layout()
        stamp_figure(fig)
        self.canvas.draw_idle()
