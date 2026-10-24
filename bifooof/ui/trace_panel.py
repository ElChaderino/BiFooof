"""Paged multichannel trace viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QScrollBar,
    QVBoxLayout, QWidget, QDoubleSpinBox, QPushButton,
)

from ..io.recording import Recording
from ..jobs import display_envelope
from ..analysis.display import display_filter
from .layouts import FlowLayout, control_group
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
    interval_selected = Signal(float, float)

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
        self.selection = None
        self.fit_window = None
        self.overlays = []
        self._drag = None

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
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self.jump = QDoubleSpinBox()
        self.jump.setRange(0, 1e8)
        self.jump.setSuffix(" s")
        go = QPushButton("Go")
        go.setObjectName("Secondary")
        go.clicked.connect(lambda: self.show_time(self.jump.value()))
        self.display_band = QComboBox()
        self.display_band.addItems(["Off", "0.5–40", "1–30", "1–70"])
        self.notch = QComboBox()
        for label, value in (("Off", 0), ("50 Hz", 50), ("60 Hz", 60)):
            self.notch.addItem(label, value)
        self.montage = QComboBox()
        self.montage.addItems(["As stored", "Average display", "Bipolar display"])
        self.display_band.currentIndexChanged.connect(self._render)
        self.notch.currentIndexChanged.connect(self._render)
        self.montage.currentIndexChanged.connect(self._render)
        tools = QWidget()
        flow = FlowLayout(tools)
        for label, widgets in (("JUMP", (self.jump, go)), ("DISPLAY BAND Hz", (self.display_band,)),
                               ("NOTCH", (self.notch,)), ("MONTAGE", (self.montage,))):
            flow.addWidget(control_group(label, *widgets))
        self.ruler = QLabel("Drag to select an interval and measure time/amplitude. Display filters do not alter fits.")
        self.ruler.setWordWrap(True)
        self.ruler.setObjectName("Muted")

        self.canvas = MplCanvas(width=9.0, height=5.2)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.setEnabled(False)
        self.scroll.valueChanged.connect(self._on_scroll)

        right = QVBoxLayout()
        right.addLayout(controls)
        right.addWidget(tools)
        right.addWidget(self.ruler)
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
        self._shortcuts = []
        for keys, direction in (("Left", -1), ("Right", 1), ("PgUp", -1), ("PgDown", 1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda value=direction: self._on_scroll(self.scroll.value() + value * self.scroll.pageStep()))
            self._shortcuts.append(shortcut)
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
        self._configure_scroll()
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
        if event.xdata is not None:
            self._drag = (float(event.xdata), nearest)
        self.channel_selected.emit(self._visible[nearest])

    def _on_release(self, event):
        if self._drag is None or event.xdata is None or event.ydata is None:
            return
        start, row = self._drag
        self._drag = None
        end = float(event.xdata)
        if abs(end - start) < .02:
            return
        left, right = sorted((start, end))
        self.selection = (left, right)
        times, values, names = self._ruler_data
        indices = [min(len(times) - 1, max(0, int(np.searchsorted(times, moment)))) for moment in (start, end)]
        delta_uv = values[row, indices[1]] - values[row, indices[0]]
        amplitude = f"{delta_uv:+.2f} µV" if np.isfinite(delta_uv) else "unavailable across excluded samples"
        self.ruler.setText(f"Selected {left:.3f}–{right:.3f} s · Δt {right-left:.3f} s · {names[row]} displayed-sample Δamplitude {amplitude}")
        self.interval_selected.emit(left, right)
        self._render()

    def _empty(self, message: str) -> None:
        self._visible = []
        self._drag = None
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
        all_data = np.array(recording.data[:, start:stop], copy=True)
        if recording.valid is not None:
            all_data[~recording.valid[:, start:stop]] = np.nan
        if self.montage.currentIndex() == 1:
            valid = np.isfinite(all_data)
            eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
            valid &= eligible[:, None]
            count_valid = valid.sum(axis=0)
            mean = np.divide(np.where(valid, all_data, 0).sum(axis=0), count_valid, out=np.zeros(all_data.shape[1]), where=count_valid >= 2)
            all_data[eligible] -= mean
        names = [recording.ch_names[index] for index in visible]
        segment = all_data[visible]
        if self.montage.currentIndex() == 2:
            pairs = [(a, b) for a, b in (("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
                ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"), ("Fz", "Cz"), ("Cz", "Pz"))
                if a in names and b in names]
            if not pairs:
                self._empty("No supported bipolar pair is visible. Choose As stored or show both channels of a pair.")
                return
            segment = np.vstack([all_data[recording.ch_names.index(a)] - all_data[recording.ch_names.index(b)] for a, b in pairs])
            names = [f"{a}−{b}" for a, b in pairs]
            visible = [recording.ch_names.index(a) for a, b in pairs]
            self._visible = visible
        try:
            segment = display_filter(segment, sfreq, self.display_band.currentText(), self.notch.currentData())
        except ValueError as exc:
            self._empty(str(exc))
            return

        self._ruler_data = (times, segment, names)

        fig = self.canvas.fig
        clear_figure(fig)
        ax = fig.add_subplot(111)
        separation = 1.0
        scale = separation / (2.0 * self._gain)
        count = segment.shape[0]
        offsets = [(count - 1 - row) * separation for row in range(count)]
        for offset in offsets:
            ax.axhline(offset, color="#e4eaf1", lw=0.6, zorder=0)
        if self.fit_window:
            ax.axvspan(*self.fit_window, color="#1c4e78", alpha=.06, zorder=0)
        if self.selection:
            ax.axvspan(*self.selection, color="#0e8f86", alpha=.13, zorder=1)
        for item in self.overlays:
            left = float(item.get("time_s", item.get("start_s", 0)))
            right = float(item.get("end_s", left + float(item.get("duration_s", 0))))
            if right < self._t0 or left > self._t0 + self._win_s:
                continue
            ax.axvspan(left, max(right, left + .02), color="#9a3412" if item.get("excluded") else "#63748a", alpha=.15)
        for row, channel_index in enumerate(visible):
            color = _SELECTED if channel_index == self._selected else _TRACE
            width = 1.15 if channel_index == self._selected else 0.7
            clipped = np.clip(segment[row] * scale, -separation, separation)
            points = display_envelope(segment[row])
            ax.plot(times[points], clipped[points] + offsets[row], color=color, lw=width, zorder=2)

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
        ax.text(.99, .99, f"DISPLAY: {self.montage.currentText()} · band {self.display_band.currentText()} · notch {self.notch.currentText()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#63748a")
        self.canvas.draw_idle()
