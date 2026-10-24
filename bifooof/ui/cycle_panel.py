"""Cycle view: band-limited trace, bursts, and bycycle shape features."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QScrollArea,
    QScrollBar, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..analysis.cycles import BANDS, CycleFit, fit_cycles
from ..analysis.settings import FitSettings
from ..io.recording import Recording
from ..jobs import display_envelope
from ..plots import clear_figure, stamp_figure
from .canvas import MplCanvas, style_axes
from .layouts import CompactDoubleSpinBox
from .workers import CallWorker, launch, wait_for

_BURST = "#0e8f86"
_TRACE = "#16324f"


class CyclePanel(QWidget):
    status = Signal(str)
    fit_ready = Signal()

    def __init__(self):
        super().__init__()
        self.recording: Recording | None = None
        self.fit: CycleFit | None = None
        self._selected = 0
        self._t0 = 0.0
        self._win_s = 10.0
        self._generation = 0
        self._worker: CallWorker | None = None
        self.settings = FitSettings()

        self.band = QComboBox()
        for label, limits in BANDS:
            self.band.addItem(label, limits)
        self.band.addItem("Custom", None)
        self.band.setCurrentIndex(2)
        self.band.setToolTip(
            "Same band as the fit bar. Alpha is the usual eyes-closed band. "
            "Delta, theta, beta, gamma, or a custom range refit bycycle and the comparison."
        )
        self.low = _hertz(8.0)
        self.high = _hertz(12.0)
        self.low.setToolTip("Low edge of a custom cycle band, in Hz.")
        self.high.setToolTip("High edge of a custom cycle band, in Hz.")
        self.band.currentIndexChanged.connect(self._on_band)
        self.low.valueChanged.connect(self._on_custom_band)
        self.high.valueChanged.connect(self._on_custom_band)

        self.caption = QLabel("The fit bar chooses the band. Bursts are shaded teal.")
        self.caption.setObjectName("Muted")
        self.caption.setWordWrap(True)
        controls = QHBoxLayout()
        band_label = QLabel("Band")
        band_label.setObjectName("Muted")
        controls.addWidget(band_label)
        controls.addWidget(self.band)
        controls.addWidget(self.low)
        controls.addWidget(self.high)
        hz = QLabel("Hz")
        hz.setObjectName("Muted")
        controls.addWidget(hz)
        controls.addStretch(1)

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
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setAlternatingRowColors(True)

        self.cutoffs = QTableWidget(4, 4)
        self.cutoffs.setHorizontalHeaderLabels(["Burst cutoff", "Needs above", "Median", "Passed"])
        self.cutoffs.verticalHeader().setVisible(False)
        self.cutoffs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cutoffs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cutoffs.setWordWrap(False)
        self.cutoffs.verticalHeader().setDefaultSectionSize(28)
        self.cutoffs.setAlternatingRowColors(True)
        self.cutoffs.setToolTip(
            "A cycle passes when its value is strictly above the cutoff. "
            "bycycle then drops the first and last cycle, and drops any run shorter than the minimum."
        )

        self.episodes = QTableWidget(0, 5)
        self.episodes.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Cycles", "Period (ms)", "Voltage (µV)"])
        self.episodes.verticalHeader().setVisible(False)
        self.episodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.episodes.setEditTriggers(QTableWidget.NoEditTriggers)
        self.episodes.setWordWrap(False)
        self.episodes.verticalHeader().setDefaultSectionSize(28)
        self.episodes.setAlternatingRowColors(True)
        self.episodes.setToolTip("Each row is one contiguous burst. A burst cannot cross a rejected stretch.")

        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.scroll)
        tables = QWidget()
        table_layout = QVBoxLayout(tables)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(6)
        table_layout.addWidget(self.table)
        table_layout.addWidget(self.cutoffs)
        table_layout.addWidget(self.episodes)
        table_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tables)
        self._fit_table(self.table, 8)
        self._fit_table(self.cutoffs, 8)
        self._fit_table(self.episodes, 6)
        self._split = QSplitter(Qt.Vertical)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(plot)
        self._split.addWidget(scroll)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        self._split_ready = False
        self.canvas.setMinimumHeight(220)
        root = QVBoxLayout(self)
        root.addLayout(controls)
        root.addWidget(self.caption)
        root.addWidget(self._split, 1)
        self._empty("Open a recording, then pick a channel.")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._give_plot_the_room()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._give_plot_the_room()

    def _give_plot_the_room(self) -> None:
        height = self._split.height()
        if height < 240 or self._split_ready:
            return
        self._split_ready = True
        self._split.setSizes([int(height * 0.52), int(height * 0.48)])

    def set_recording(self, recording: Recording, selected: int | None = None) -> None:
        self.recording = recording
        self.fit = None
        self._t0 = 0.0
        if selected is None:
            self._selected = 0
        else:
            self._selected = max(0, min(int(selected), recording.n_channels - 1))
        self._refit()

    def current_band(self) -> tuple[float, float]:
        return float(self.low.value()), float(self.high.value())

    def set_band(self, low: float, high: float, *, named: bool) -> None:
        """Match the fit bar. Signals stay blocked so this does not echo or refit twice."""
        same = (
            round(float(self.low.value()), 4) == round(float(low), 4)
            and round(float(self.high.value()), 4) == round(float(high), 4)
        )
        self.band.blockSignals(True)
        self.low.blockSignals(True)
        self.high.blockSignals(True)
        if named:
            matched = False
            for index in range(self.band.count()):
                data = self.band.itemData(index)
                if data is not None and abs(float(data[0]) - low) < .00005 and abs(float(data[1]) - high) < .00005:
                    self.band.setCurrentIndex(index)
                    matched = True
                    break
            if not matched:
                custom = self.band.findData(None)
                if custom >= 0:
                    self.band.setCurrentIndex(custom)
        else:
            custom = self.band.findData(None)
            if custom >= 0:
                self.band.setCurrentIndex(custom)
        self.low.setValue(float(low))
        self.high.setValue(float(high))
        self.band.blockSignals(False)
        self.low.blockSignals(False)
        self.high.blockSignals(False)
        if not same:
            self._refit()

    def apply_settings(self, settings: FitSettings) -> None:
        self.settings = replace(settings)
        self._t0 = 0.0
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
        band = self.current_band()
        self._generation += 1
        self.fit = None
        self.fit_ready.emit()
        name = self.recording.ch_names[index]
        self.caption.setText(f"Finding cycles on {name}…")
        self.status.emit(f"Finding cycles on {name}…")
        worker = CallWorker(self._generation, fit_cycles, (self.recording, index, band, replace(self.settings)))
        worker.succeeded.connect(self._fitted)
        worker.failed.connect(self._failed)
        self._worker = worker
        launch(self, worker)

    def _on_band(self, _index: int = 0) -> None:
        data = self.band.currentData()
        if data is not None:
            self.low.blockSignals(True)
            self.high.blockSignals(True)
            self.low.setValue(float(data[0]))
            self.high.setValue(float(data[1]))
            self.low.blockSignals(False)
            self.high.blockSignals(False)
        self._refit()

    def _on_custom_band(self, _value: float = 0.0) -> None:
        custom = self.band.findData(None)
        if custom >= 0 and self.band.currentIndex() != custom:
            self.band.blockSignals(True)
            self.band.setCurrentIndex(custom)
            self.band.blockSignals(False)
        self._refit()

    def _fitted(self, generation: int, result: object) -> None:
        if generation != self._generation or not isinstance(result, CycleFit):
            return
        self.fit = result
        self._configure_scroll()
        self._fill_table()
        self._draw()
        share = ""
        if result.burst_fraction is not None:
            share = (
                f", {result.burst_duration_s:.2f} s "
                f"({100.0 * result.burst_fraction:.1f}% of accepted time)"
            )
        text = (
            f"{result.channel}  ·  {result.band[0]:g}–{result.band[1]:g} Hz  ·  "
            f"{result.n_cycles} cycles, {result.n_burst_cycles} burst cycles "
            f"in {result.n_burst_episodes} episodes{share}"
        )
        if result.burst_note:
            text = f"{text}. {result.burst_note}"
        if result.notes:
            text = f"{text}. {result.notes}"
        if result.failures:
            text += f". {len(result.failures)} accepted run(s) could not be fitted; see details."
        self.caption.setText(text)
        self.caption.setToolTip("\n".join(f"{item['start_s']:.3f}–{item['end_s']:.3f} s: {item['reason']}" for item in result.failures))
        self.status.emit("Cycle fit ready.")
        self.fit_ready.emit()

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

    def show_time(self, t_file: float) -> None:
        """Scroll the filtered trace to a time on the recording clock."""
        if self.fit is None or self.fit.sfreq <= 0:
            return
        local = float(t_file) - float(self.fit.t_offset)
        duration = len(self.fit.filtered) / self.fit.sfreq
        max_t0 = max(0.0, duration - self._win_s)
        self._t0 = min(max(0.0, local - self._win_s / 2.0), max_t0)
        self._configure_scroll()
        self._draw()

    def _fit_table(self, table: QTableWidget, max_rows: int) -> None:
        """Give the table the height of its rows so the next table cannot draw over it."""
        table.resizeRowsToContents()
        shown = min(table.rowCount(), max_rows)
        height = table.horizontalHeader().sizeHint().height() + table.frameWidth() * 2 + 8
        if shown <= 0:
            height += 32
        else:
            height += sum(max(table.rowHeight(index), 26) for index in range(shown))
        table.setFixedHeight(height)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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
        self._fit_table(self.table, 8)
        self._fill_cutoffs()
        self._fill_episodes()

    def _fill_cutoffs(self) -> None:
        fit = self.fit
        rows = [] if fit is None else fit.cutoffs
        self.cutoffs.setRowCount(max(len(rows), 1))
        if not rows:
            self.cutoffs.setItem(0, 0, _cell("—"))
            for column in range(1, 4):
                self.cutoffs.setItem(0, column, _cell(""))
            self._fit_table(self.cutoffs, 8)
            return
        for row, cutoff in enumerate(rows):
            self.cutoffs.setItem(row, 0, _cell(cutoff.label.capitalize()))
            self.cutoffs.setItem(row, 1, _cell(f"{cutoff.cutoff:g}"))
            self.cutoffs.setItem(row, 2, _cell(_num(cutoff.median)))
            self.cutoffs.setItem(row, 3, _cell(f"{cutoff.n_passed} / {cutoff.n_cycles}"))
        self._fit_table(self.cutoffs, 8)

    def _fill_episodes(self) -> None:
        fit = self.fit
        rows = [] if fit is None else fit.episodes
        self.episodes.setRowCount(max(len(rows), 1))
        if not rows:
            self.episodes.setItem(0, 0, _cell("No burst episodes"))
            for column in range(1, 5):
                self.episodes.setItem(0, column, _cell(""))
            self._fit_table(self.episodes, 6)
            return
        for row, episode in enumerate(rows):
            self.episodes.setItem(row, 0, _cell(f"{episode.start_s:.3f}"))
            self.episodes.setItem(row, 1, _cell(f"{episode.end_s:.3f}"))
            self.episodes.setItem(row, 2, _cell(str(episode.n_cycles)))
            self.episodes.setItem(row, 3, _cell(_num(episode.period_ms)))
            self.episodes.setItem(row, 4, _cell(_num(episode.volt_amp)))
        self._fit_table(self.episodes, 6)

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
        times = fit.t_offset + np.arange(start, stop) / sfreq
        values = fit.filtered[start:stop]
        if not np.isfinite(values).any():
            self._empty("No accepted filtered samples in this page. Inspect exclusions or select another channel.")
            return
        points = display_envelope(values)
        view_t = times[points]
        view_y = values[points]

        fig = self.canvas.fig
        clear_figure(fig)
        ax = fig.add_subplot(111)
        t_end = times[-1]
        for span_start, span_stop, is_burst in fit.spans:
            if not is_burst:
                continue
            left = fit.t_offset + span_start / sfreq
            right = fit.t_offset + span_stop / sfreq
            if right < times[0] or left > t_end:
                continue
            ax.axvspan(max(left, times[0]), min(right, t_end), color=_BURST, alpha=0.28, lw=0, zorder=0)
        peak_idx = [peak for peak in fit.peaks if start <= peak < stop]
        if peak_idx:
            ax.plot(fit.t_offset + np.asarray(peak_idx) / sfreq, fit.filtered[peak_idx],
                    linestyle="none", marker="o", ms=3.5, color="#b3261e", label="Peaks", zorder=3)
        ax.plot(view_t, view_y, color=_TRACE, lw=0.95, label="Filtered", zorder=2)
        pad = max(2.0, float(np.nanmax(np.abs(view_y))) * 0.12)
        ax.set_ylim(float(np.nanmin(view_y)) - pad, float(np.nanmax(view_y)) + pad)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV")
        ax.set_title(
            f"{fit.channel} filtered {fit.band[0]:g}–{fit.band[1]:g} Hz"
            "    ·    teal = burst",
            fontsize=9, loc="left", color="#16232f",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style_axes(ax)
        fig.tight_layout()
        stamp_figure(fig)
        self.canvas.draw_idle()


def _hertz(value: float) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox()
    box.setRange(0.5, 80.0)
    box.setDecimals(4)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setFixedWidth(76)
    return box


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignCenter)
    return item


def _num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"
