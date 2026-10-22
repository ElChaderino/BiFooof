"""Spectrum view: specparam overlay, parameter table, and 10-20 maps."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..analysis.settings import FitSettings
from ..analysis.specparam_fit import SpectrumFit, fit_spectra
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from ..plots import render_spectrum
from .canvas import MplCanvas
from .workers import CallWorker, launch, wait_for

_COLUMNS = [
    "Channel", "Offset", "Exponent", "CF (Hz)", "Power",
    "BW (Hz)", "Peaks", "R²", "Error",
]


class SpectrumPanel(QWidget):
    channel_selected = Signal(int)
    status = Signal(str)
    fit_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.recording: Recording | None = None
        self.recordings: dict[str, Recording] = {}
        self.fit: SpectrumFit | None = None
        self.fits: dict[str, SpectrumFit] = {}
        self.condition = "ec"
        self._selected = 0
        self._generations: dict[str, int] = {}
        self._workers: dict[str, CallWorker] = {}
        self._worker: CallWorker | None = None
        self.settings = FitSettings()

        self.caption = QLabel("Welch spectrum, 1–40 Hz, with the specparam aperiodic fit and peaks.")
        self.caption.setObjectName("Muted")
        self.caption.setWordWrap(True)
        self.canvas = MplCanvas(width=8.0, height=6.0)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(520)
        self.table.setMaximumWidth(680)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_table)

        plots = QVBoxLayout()
        plots.addWidget(self.caption)
        plots.addWidget(self.canvas, 1)
        root = QHBoxLayout(self)
        root.addLayout(plots, 1)
        root.addWidget(self.table)
        self._empty("The spectrum fit starts after a recording is opened.")

    def set_recording(self, recording: Recording, condition: str = "ec") -> None:
        self.recordings[condition] = recording
        self.condition = condition
        self.recording = recording
        self.fits.pop(condition, None)
        self.fit = None
        self._selected = 0
        self.table.setRowCount(0)
        self._empty("Fitting spectra…")
        self._start(condition)

    def show_condition(self, condition: str) -> None:
        """Display a fit that is already running or finished, without starting another."""
        self.condition = condition
        self.recording = self.recordings.get(condition)
        self.fit = self.fits.get(condition)
        if self.fit is None:
            self.table.setRowCount(0)
            label = "Eyes closed" if condition == "ec" else "Eyes open"
            self._empty(f"Fitting {label.lower()} spectra…")
            return
        self._fill_table()
        self._draw()

    def apply_settings(self, settings: FitSettings) -> None:
        self.settings = settings
        self.fits.clear()
        self.fit = None
        self.table.setRowCount(0)
        if not self.recordings:
            return
        self._empty("Fitting spectra…")
        for condition in list(self.recordings):
            self._start(condition)

    def set_selected(self, index: int) -> None:
        if self.fit is None or not 0 <= index < len(self.fit.channels):
            self._selected = index
            return
        self._selected = index
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self._draw()

    def shutdown(self) -> None:
        wait_for(self)

    def _start(self, condition: str) -> None:
        recording = self.recordings.get(condition)
        if recording is None:
            return
        self._generations[condition] = self._generations.get(condition, 0) + 1
        generation = self._generations[condition]
        label = "eyes closed" if condition == "ec" else "eyes open"
        self.status.emit(f"Fitting {label} spectra…")
        worker = CallWorker(generation, fit_spectra, (recording, self.settings))
        worker.succeeded.connect(lambda gen, result, cond=condition: self._fitted(cond, gen, result))
        worker.failed.connect(lambda gen, message, cond=condition: self._failed(cond, gen, message))
        self._workers[condition] = worker
        self._worker = worker
        launch(self, worker)

    def _fitted(self, condition: str, generation: int, result: object) -> None:
        if generation != self._generations.get(condition) or not isinstance(result, SpectrumFit):
            return
        self.fits[condition] = result
        self.fit_ready.emit(condition)
        if condition != self.condition:
            self.status.emit("Spectrum fit ready.")
            return
        self.fit = result
        self._fill_table()
        self._draw()
        self.status.emit("Spectrum fit ready.")

    def _failed(self, condition: str, generation: int, message: str) -> None:
        if generation != self._generations.get(condition):
            return
        detail = message.strip().splitlines()[-1] if message.strip() else "Spectrum fit failed."
        self.status.emit(detail)
        if condition == self.condition:
            self.fit = None
            self._empty(detail)

    def _fill_table(self) -> None:
        assert self.fit is not None
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.fit.channels))
        for row, item in enumerate(self.fit.channels):
            peak = item.dominant_peak
            centers = ", ".join(f"{cf:.1f}" for cf, _pw, _bw in item.peaks) if item.peaks else "—"
            values = [
                item.name,
                _num(item.offset, 3),
                _num(item.exponent, 3),
                _num(peak[0], 2) if peak else "—",
                _num(peak[1], 3) if peak else "—",
                _num(peak[2], 2) if peak else "—",
                centers if item.usable else (item.reason or "—"),
                _num(item.r_squared, 3),
                _num(item.error, 4),
            ]
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, cell)
        if self.fit.channels:
            self.table.selectRow(min(self._selected, len(self.fit.channels) - 1))
        self.table.blockSignals(False)

    def _on_table(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        index = rows[0].row()
        self._selected = index
        self._draw()
        self.channel_selected.emit(index)

    def _empty(self, message: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="#63748a", fontsize=12)
        self.canvas.draw_idle()

    def _draw(self) -> None:
        if self.fit is None:
            return
        render_spectrum(
            self.canvas.fig, self.fit, self._selected,
            CONDITION_LABELS.get(self.condition, ""),
        )
        self._describe()
        self.canvas.draw_idle()

    def _describe(self) -> None:
        if self.fit is None or not self.fit.channels:
            return
        index = min(max(self._selected, 0), len(self.fit.channels) - 1)
        item = self.fit.channels[index]
        label = CONDITION_LABELS.get(self.condition, self.condition)
        if not item.usable:
            self.caption.setText(f"{label}  ·  {item.name}  ·  {item.reason or 'Not fit'}")
            return
        peak = "no peak" if item.dominant_peak is None else f"peak {item.dominant_peak[0]:.1f} Hz"
        exponent = "—" if item.exponent is None else f"exponent {item.exponent:.2f}"
        r_squared = "—" if item.r_squared is None else f"R² {item.r_squared:.2f}"
        knee = ""
        if item.knee is not None and item.exponent not in (None, 0):
            knee_hz = item.knee ** (1.0 / item.exponent)
            if np.isfinite(knee_hz) and knee_hz > 0:
                knee = f"  ·  knee {knee_hz:.1f} Hz"
        window = f"{self.fit.t_start:.0f}–{self.fit.t_end:.0f} s"
        span = f"{self.fit.fmin:g}–{self.fit.fmax:.0f} Hz"
        self.caption.setText(
            f"{label}  ·  {item.name}  ·  {window}  ·  {span}  ·  "
            f"{self.fit.aperiodic_mode}{knee}  ·  {exponent}  ·  {peak}  ·  {r_squared}"
        )


def _num(value: float | None, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"
