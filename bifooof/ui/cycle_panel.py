"""Cycle view: band-limited trace, bursts, and bycycle shape features."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QScrollBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..analysis.cycles import BANDS, CycleFit, fit_cycles
from ..io.recording import Recording
from .canvas import MplCanvas, style_axes
from .workers import CallWorker, launch, wait_for

_BURST = "#0e8f86"
_QUIET = "#d5dde6"
_TRACE = "#16324f"


class CyclePanel(QWidget):
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.recording: Recording | None = None
        self.fit: CycleFit | None = None
        self._selected = 0
        self._t0 = 0.0
        self._win_s = 10.0
        self._generation = 0
        self._worker: CallWorker | None = None

        self.band = QComboBox()
        for label, limits in BANDS:
            self.band.addItem(label, limits)
        self.band.setCurrentIndex(2)
        self.band.setToolTip("Frequency band passed to bycycle. Alpha is the usual eyes-closed band.")
        self.band.currentIndexChanged.connect(self._refit)

        self.caption = QLabel("Alpha is the default band. Bursts are shaded teal.")
        self.caption.setObjectName("Muted")
        controls = QHBoxLayout()
        band_label = QLabel("Band")
        band_label.setObjectName("Muted")
        controls.addWidget(band_label)
        controls.addWidget(self.band)
        controls.addWidget(self.caption, 1)

        self.canvas = MplCanvas(width=9.0, height=4.2)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.setEnabled(False)
        self.scroll.valueChanged.connect(self._on_scroll)

        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["Measure", "All cycles", "Bursts"])
        self.table.setVerticalHeaderLabels(["Period (ms)", "Rise–decay", "Peak–trough", "Voltage (µV)"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(170)
        self.table.setAlternatingRowColors(True)

        root = QVBoxLayout(self)
        root.addLayout(controls)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.scroll)
        root.addWidget(self.table)
        self._empty("Open a recording, then pick a channel.")

    def set_recording(self, recording: Recording, selected: int | None = None) -> None:
        self.recording = recording
        self.fit = None
        self._t0 = 0.0
        if selected is None:
            self._selected = 0
        else:
            self._selected = max(0, min(int(selected), recording.n_channels - 1))
        self._refit()

    def set_selected(self, index: int) -> None:
        if index == self._selected and self.fit is not None and self.fit.channel_index == index:
            return
        self._selected = index
        self._t0 = 0.0
        self._refit()

    def shutdown(self) -> None:
        wait_for(self)

    def _refit(self, _index: int = 0) -> None:
        if self.recording is None or self.recording.n_channels == 0:
            return
        index = min(max(self._selected, 0), self.recording.n_channels - 1)
        band = self.band.currentData()
        self._generation += 1
        name = self.recording.ch_names[index]
        self.caption.setText(f"Finding cycles on {name}…")
        self.status.emit(f"Finding cycles on {name}…")
        worker = CallWorker(self._generation, fit_cycles, (self.recording, index, band))
        worker.succeeded.connect(self._fitted)
        worker.failed.connect(self._failed)
        self._worker = worker
        launch(self, worker)

    def _fitted(self, generation: int, result: object) -> None:
        if generation != self._generation or not isinstance(result, CycleFit):
            return
        self.fit = result
        self._configure_scroll()
        self._fill_table()
        self._draw()
        self.caption.setText(
            f"{result.channel}  ·  {result.band[0]:g}–{result.band[1]:g} Hz  ·  "
            f"{result.n_cycles} cycles, {result.n_bursts} in bursts"
        )
        self.status.emit("Cycle fit ready.")

    def _failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.fit = None
        detail = message.strip().splitlines()[-1] if message.strip() else "Cycle fit failed."
        self.caption.setText(detail)
        self._empty(detail)
        self._fill_table()
        self.status.emit(detail)

    def _configure_scroll(self) -> None:
        duration = 0.0 if self.fit is None else len(self.fit.filtered) / self.fit.sfreq
        max_t0 = max(0.0, duration - self._win_s)
        self._t0 = min(self._t0, max_t0)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, int(round(max_t0 * 10)))
        self.scroll.setPageStep(int(round(self._win_s * 10)))
        self.scroll.setSingleStep(10)
        self.scroll.setValue(int(round(self._t0 * 10)))
        self.scroll.setEnabled(max_t0 > 0)
        self.scroll.blockSignals(False)

    def _on_scroll(self, value: int) -> None:
        self._t0 = value / 10.0
        self._draw()

    def _fill_table(self) -> None:
        fit = self.fit
        rows = [
            ("Period (ms)", None if fit is None else fit.period_ms, None if fit is None else fit.burst_period_ms),
            ("Rise–decay symmetry", None if fit is None else fit.rise_decay, None if fit is None else fit.burst_rise_decay),
            ("Peak–trough symmetry", None if fit is None else fit.peak_trough, None if fit is None else fit.burst_peak_trough),
            ("Voltage amplitude (µV)", None if fit is None else fit.volt_amp, None if fit is None else fit.burst_volt_amp),
        ]
        for row, (label, all_cycles, bursts) in enumerate(rows):
            self.table.setItem(row, 0, _cell(label))
            self.table.setItem(row, 1, _cell(_num(all_cycles)))
            self.table.setItem(row, 2, _cell(_num(bursts)))

    def _empty(self, message: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#63748a", fontsize=12)
        self.canvas.draw_idle()

    def _draw(self) -> None:
        fit = self.fit
        if fit is None or fit.filtered.size == 0:
            return
        sfreq = fit.sfreq
        start = int(round(self._t0 * sfreq))
        stop = min(int(round((self._t0 + self._win_s) * sfreq)), fit.filtered.size)
        if stop <= start:
            self._empty("This page has no samples.")
            return
        times = np.arange(start, stop) / sfreq
        step = max(1, len(times) // 2500)
        view_t = times[::step]
        view_y = fit.filtered[start:stop][::step]

        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        t_end = times[-1]
        for span_start, span_stop, is_burst in fit.spans:
            left = span_start / sfreq
            right = span_stop / sfreq
            if right < times[0] or left > t_end:
                continue
            ax.axvspan(max(left, times[0]), min(right, t_end),
                       color=_BURST if is_burst else _QUIET, alpha=0.45, lw=0)
        peak_idx = [peak for peak in fit.peaks if start <= peak < stop]
        if peak_idx:
            ax.plot(np.asarray(peak_idx) / sfreq, fit.filtered[peak_idx],
                    linestyle="none", marker="o", ms=3.5, color="#b3261e", label="Peaks")
        ax.plot(view_t, view_y, color=_TRACE, lw=0.9, label="Filtered")
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV")
        ax.set_title(
            f"{fit.channel} filtered {fit.band[0]:g}–{fit.band[1]:g} Hz"
            "    ·    teal = burst, gray = other cycles",
            fontsize=9, loc="left", color="#16232f",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style_axes(ax)
        fig.tight_layout()
        self.canvas.draw_idle()


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"
