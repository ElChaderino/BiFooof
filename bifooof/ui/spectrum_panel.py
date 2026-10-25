"""Spectrum view: specparam overlay, parameter table, and 10-20 maps."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..analysis.compare import alpha_peak
from ..analysis.prepare import R2_WEAK, peak_at_width_floor
from ..analysis.settings import FitSettings, format_hz, format_seconds, matched_settings, reference_phrase
from ..analysis.specparam_fit import SpectrumFit, epochs_for_peak, fit_spectra
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from ..plots import render_spectrum
from .canvas import MplCanvas
from .workers import CallWorker, launch, wait_for

_COLUMNS = [
    "Channel", "Offset", "Exponent", "CF (Hz)", "Height",
    "BW (Hz)", "Peaks", "R²", "Error",
]


class SpectrumPanel(QWidget):
    channel_selected = Signal(int)
    status = Signal(str)
    fit_ready = Signal(str)
    fit_invalidated = Signal(str)

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
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._mark_hz: float | None = None
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        cf_header = self.table.horizontalHeaderItem(3)
        if cf_header is not None:
            cf_header.setToolTip(
                "Center of the tallest fitted peak in the whole fit range. "
                "It can sit outside the selected band. The caption names the in-band peak separately."
            )
        height_header = self.table.horizontalHeaderItem(4)
        if height_header is not None:
            height_header.setToolTip(
                "Height of the tallest fitted peak above the aperiodic background, in log10 power. "
                "Not mean Welch power in a band, and not a share of the spectrum."
            )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(460)
        self.table.setMaximumWidth(560)
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
        if len(self.recordings) > 1:
            for other in list(self.recordings):
                if other != condition:
                    self._start(other)

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
        self.settings = replace(settings)
        self.fits.clear()
        self.fit = None
        self.table.setRowCount(0)
        if not self.recordings:
            return
        self._empty("Fitting spectra…")
        for condition in list(self.recordings):
            self._start(condition)

    def set_band(self, low: float, high: float) -> None:
        """The band changes the in-band peak without refitting specparam."""
        self.settings = replace(self.settings, alpha_low=float(low), alpha_high=float(high))
        for fit in self.fits.values():
            fit.settings = replace(fit.settings or self.settings, alpha_low=float(low), alpha_high=float(high))
        if self.fit is not None:
            self._fill_table()
            self._describe()

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
        self.fits.pop(condition, None)
        if condition == self.condition:
            self.fit = None
            self.table.setRowCount(0)
            self._empty("Fitting spectra…")
        self.fit_invalidated.emit(condition)
        label = "eyes closed" if condition == "ec" else "eyes open"
        self.status.emit(f"Fitting {label} spectra…")
        others = [item for key, item in self.recordings.items() if key != condition]
        settings = replace(matched_settings(self.settings, recording, others))
        worker = CallWorker(generation, fit_spectra, (recording, settings))
        worker.succeeded.connect(lambda gen, result, cond=condition: self._fitted(cond, gen, result))
        worker.failed.connect(lambda gen, message, cond=condition: self._failed(cond, gen, message))
        self._workers[condition] = worker
        self._worker = worker
        launch(self, worker, key=condition)

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
        self.fits.pop(condition, None)
        self.fit_invalidated.emit(condition)
        detail = message.strip().splitlines()[-1] if message.strip() else "Spectrum fit failed."
        self.status.emit(detail)
        if condition == self.condition:
            self.fit = None
            self.table.setRowCount(0)
            self._empty(detail)

    def _fill_table(self) -> None:
        assert self.fit is not None
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.fit.channels))
        for row, item in enumerate(self.fit.channels):
            peak = item.dominant_peak
            centers = ", ".join(f"{cf:.1f}" for cf, _pw, _bw in item.peaks) if item.peaks else "—"
            bandwidth = "—"
            if peak:
                bandwidth = f"{peak[2]:.2f}"
                if peak_at_width_floor(peak[2], self.settings.peak_width_low):
                    bandwidth = f"{bandwidth} floor"
            values = [
                item.name,
                _num(item.offset, 3),
                _num(item.exponent, 3),
                _num(peak[0], 2) if peak else "—",
                _num(peak[1], 3) if peak else "—",
                bandwidth,
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

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if event.inaxes.get_label() not in {"spectrum", "residual"}:
            return
        self._mark_hz = float(event.xdata)
        self._draw()

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
            self._mark_hz,
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
        band_peak = alpha_peak(item, self.settings.alpha_band())
        tallest = item.dominant_peak
        if band_peak is None:
            peak = "no in-band peak" if tallest is None else f"no in-band peak, tallest {tallest[0]:.1f} Hz"
        elif tallest is not None and abs(tallest[0] - band_peak[0]) > 0.05:
            peak = f"in-band {band_peak[0]:.1f} Hz, tallest {tallest[0]:.1f} Hz"
        else:
            peak = f"in-band peak {band_peak[0]:.1f} Hz"
        exponent = "—" if item.exponent is None else f"exponent {item.exponent:.2f}"
        r_squared = "—" if item.r_squared is None else f"R² {item.r_squared:.2f}"
        knee = ""
        if item.knee is not None and item.knee > 0 and item.exponent not in (None, 0):
            with np.errstate(over="ignore", invalid="ignore"):
                knee_hz = np.exp(np.log(item.knee) / item.exponent)
            if np.isfinite(knee_hz) and knee_hz > 0:
                knee = f"  ·  knee {knee_hz:.1f} Hz"
        window = f"{format_seconds(self.fit.t_start)}–{format_seconds(self.fit.t_end)} s"
        span = f"{self.fit.fmin:g}–{format_hz(self.fit.fmax)} Hz"
        notes = []
        if self.fit.epoch_s and item.n_epochs:
            count = epochs_for_peak(item, band_peak)
            if band_peak is not None and count:
                note = (
                    f"that in-band peak is in {count} of {item.n_epochs} "
                    f"epochs of {format_seconds(self.fit.epoch_s)} s"
                )
                if count < item.n_epochs:
                    note += ". Its height is the median of those epochs, not lowered by the epochs that missed it"
                notes.append(note)
            else:
                notes.append(f"no in-band peak in {item.n_epochs} epochs of {format_seconds(self.fit.epoch_s)} s")
        floor_peak = band_peak or tallest
        if floor_peak is not None and peak_at_width_floor(floor_peak[2], self.settings.peak_width_low):
            notes.append("peak width is at the narrowest allowed")
        if item.r_squared is not None and item.r_squared < R2_WEAK:
            notes.append("weak fit")
        if (
            self.recording is not None
            and self.settings.t_length is None
            and self.fit.t_end + 1.0 < self.recording.duration_s
        ):
            notes.append(f"matched to the shorter file, {format_seconds(self.fit.t_end)} s")
        if self.fit.notes:
            notes.append(self.fit.notes)
        if self.recording is not None and any("microvolts" in note for note in self.recording.notes):
            notes.append("check voltage")
        applied = (self.fit.provenance or {}).get("reference_applied") or self.settings.reference
        notes.append(reference_phrase(str(applied)))
        if self._mark_hz is not None and item.freqs is not None and item.spectrum is not None and item.model is not None:
            nearest = int(np.argmin(np.abs(item.freqs - self._mark_hz)))
            notes.append(
                f"residual at {item.freqs[nearest]:.1f} Hz is {item.spectrum[nearest] - item.model[nearest]:+.3f}"
            )
        suffix = "" if not notes else "  ·  " + "  ·  ".join(notes)
        self.caption.setText(
            f"{label}  ·  {item.name}  ·  {window}  ·  {span}  ·  "
            f"{self.fit.aperiodic_mode}{knee}  ·  {exponent}  ·  {peak}  ·  {r_squared}{suffix}"
        )


def _num(value: float | None, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"
