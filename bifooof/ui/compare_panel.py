"""Eyes-closed versus eyes-open comparison."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..analysis.compare import ChannelComparison, alpha_peak, compare_spectra
from ..analysis.cycles import CycleFit, fit_cycles
from ..analysis.settings import FitSettings, band_label, band_phrase
from ..analysis.specparam_fit import ChannelFit, SpectrumFit
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from ..plots import mark_spectrum, stamp_figure
from .canvas import MplCanvas, finish_figure, style_axes
from .headmap import draw_head
from .workers import CallWorker, launch, wait_for

_EC = "#1c4e78"
_EO = "#c2410c"
_LEAD = ["Channel", "Offset EC", "Offset EO", "Exp EC", "Exp EO", "EO − EC"]


class ComparePanel(QWidget):
    channel_selected = Signal(str)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.rows: list[ChannelComparison] = []
        self.recordings: dict[str, Recording] = {}
        self.fits: dict[str, SpectrumFit] = {}
        self.channel = ""
        self.band = (8.0, 12.0)
        self.settings = FitSettings()
        self.cycle_fits: dict[str, CycleFit] = {}
        self._cycle_key: tuple | None = None
        self._generations: dict[str, int] = {}
        self._workers: dict[str, CallWorker] = {}

        self.caption = QLabel(
            "Open an eyes-closed and an eyes-open recording. "
            "Reactivity is eyes-closed peak power minus eyes-open in the band on the fit bar."
        )
        self.caption.setObjectName("Muted")
        self.caption.setWordWrap(True)
        self.canvas = MplCanvas(width=8.0, height=6.0)
        self.table = QTableWidget(0, 11)
        self._refresh_headers()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(188)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_table)

        self.cycle_table = QTableWidget(4, 3)
        self.cycle_table.setHorizontalHeaderLabels(["Measure", "Eyes closed", "Eyes open"])
        self.cycle_table.verticalHeader().setVisible(False)
        self.cycle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cycle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cycle_table.setMaximumHeight(148)
        self.cycle_table.setAlternatingRowColors(True)
        self.cycle_note = QLabel("")
        self.cycle_note.setObjectName("Muted")
        self.cycle_note.setWordWrap(True)
        self.cycle_note.setVisible(False)

        root = QVBoxLayout(self)
        root.addWidget(self.caption)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.cycle_table)
        root.addWidget(self.cycle_note)
        root.addWidget(self.table)
        self._empty("Open both an eyes-closed and an eyes-open file to compare them.")
        self._fill_cycles(None, None)

    @property
    def ready(self) -> bool:
        return "ec" in self.fits and "eo" in self.fits

    def apply_settings(self, settings: FitSettings) -> None:
        self.settings = settings
        self.band = settings.alpha_band()
        self._cycle_key = None
        self._refresh_headers()
        if self.ready:
            self.rows = compare_spectra(self.fits["ec"], self.fits["eo"], self._alpha())
            self._fill_table()
            self._draw()
            self._maybe_cycles()

    def _alpha(self) -> tuple[float, float]:
        return self.settings.alpha_band()

    def set_pair(self, recordings: dict[str, Recording], fits: dict[str, SpectrumFit],
                 channel: str, band: tuple[float, float]) -> None:
        self.recordings = recordings
        self.fits = fits
        self.band = (float(band[0]), float(band[1]))
        self.settings.alpha_low = self.band[0]
        self.settings.alpha_high = self.band[1]
        self._refresh_headers()
        if not self.ready:
            self.rows = []
            self.table.setRowCount(0)
            self.cycle_fits = {}
            self._cycle_key = None
            self._fill_cycles(None, None)
            missing = [CONDITION_LABELS[key] for key in ("ec", "eo") if key not in fits]
            if {"ec", "eo"} <= set(recordings) and missing:
                self._empty("Fitting both recordings…")
            else:
                self._empty("Open both an eyes-closed and an eyes-open file to compare them.")
            return
        self.rows = compare_spectra(fits["ec"], fits["eo"], self._alpha())
        if channel and any(row.name == channel for row in self.rows):
            self.channel = channel
        elif self.rows and self.channel not in {row.name for row in self.rows}:
            self.channel = self.rows[0].name
        self._fill_table()
        self._draw()
        self._maybe_cycles()

    def shutdown(self) -> None:
        wait_for(self)

    def _refresh_headers(self) -> None:
        name = band_label(*self._alpha())
        self.table.setHorizontalHeaderLabels([
            *_LEAD,
            f"{name} EC", f"{name} EO", "EC − EO", "CF EC", "CF EO",
        ])

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        selected = 0
        for row_index, row in enumerate(self.rows):
            if row.name == self.channel:
                selected = row_index
            values = [
                row.name,
                _num(row.offset_ec, 3),
                _num(row.offset_eo, 3),
                _num(row.exponent_ec, 3),
                _num(row.exponent_eo, 3),
                _num(row.exponent_delta, 3),
                _num(row.alpha_power_ec, 3),
                _num(row.alpha_power_eo, 3),
                _num(row.alpha_reactivity, 3),
                _num(row.alpha_cf_ec, 2),
                _num(row.alpha_cf_eo, 2),
            ]
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_index, column, cell)
        if self.rows:
            self.table.selectRow(selected)
        self.table.blockSignals(False)

    def _on_table(self) -> None:
        chosen = self.table.selectionModel().selectedRows()
        if not chosen or not self.rows:
            return
        name = self.rows[chosen[0].row()].name
        if name == self.channel:
            return
        self.channel = name
        self._draw()
        self._maybe_cycles()
        self.channel_selected.emit(name)

    def _draw(self) -> None:
        if not self.rows:
            return
        fig = self.canvas.fig
        fig.clear()
        finish_figure(fig)
        grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.45])
        ax = fig.add_subplot(grid[0, :])
        ec_item = _by_name(self.fits.get("ec"), self.channel)
        eo_item = _by_name(self.fits.get("eo"), self.channel)
        fmin = min(item.fmin for item in self.fits.values())
        fmax = max(item.fmax for item in self.fits.values())
        mark_spectrum(ax, fmin, fmax)
        alpha = self._alpha()
        shown = band_phrase(*alpha)
        _plot_condition(ax, ec_item, _EC, "Eyes closed", alpha)
        _plot_condition(ax, eo_item, _EO, "Eyes open", alpha)
        ax.set_xlabel("")
        ax.set_ylabel("log10 power")
        ax.set_title(
            f"{self.channel}   ·   eyes closed vs eyes open   ·   {shown}",
            fontsize=10, loc="left",
        )
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax, grid=True)

        names = [row.name for row in self.rows]
        draw_head(
            fig.add_subplot(grid[1, 0]), names,
            [row.exponent_delta for row in self.rows],
            "Exponent  EO − EC", "viridis", self.channel,
        )
        draw_head(
            fig.add_subplot(grid[1, 1]), names,
            [row.alpha_reactivity for row in self.rows],
            f"{shown}   EC − EO", "plasma", self.channel,
        )
        stamp_figure(fig)
        self.canvas.draw_idle()
        ec_power = next((row.alpha_power_ec for row in self.rows if row.name == self.channel), None)
        eo_power = next((row.alpha_power_eo for row in self.rows if row.name == self.channel), None)
        reactivity = None if ec_power is None or eo_power is None else ec_power - eo_power
        name = band_label(*alpha)
        kind = name.lower() if name.isalpha() else name
        self.caption.setText(
            f"{self.channel}: {kind} reactivity (EC − EO) {_num(reactivity, 3)}. "
            "Positive means more power in this band with eyes closed."
        )

    def _maybe_cycles(self) -> None:
        key = (
            self.channel,
            self.band,
            self.settings.key(),
            id(self.recordings.get("ec")),
            id(self.recordings.get("eo")),
        )
        if key == self._cycle_key:
            return
        self._cycle_key = key
        self.cycle_fits = {}
        for condition in ("ec", "eo"):
            recording = self.recordings.get(condition)
            if recording is None or self.channel not in recording.ch_names:
                continue
            index = recording.ch_names.index(self.channel)
            self._generations[condition] = self._generations.get(condition, 0) + 1
            generation = self._generations[condition]
            worker = CallWorker(generation, fit_cycles, (recording, index, self.band, self.settings))
            worker.succeeded.connect(
                lambda gen, result, cond=condition: self._cycle_done(cond, gen, result)
            )
            worker.failed.connect(
                lambda gen, message, cond=condition: self._cycle_failed(cond, gen, message)
            )
            self._workers[condition] = worker
            launch(self, worker)
        self._fill_cycles(None, None)
        self.status.emit(f"Comparing cycles on {self.channel}…")

    def _cycle_done(self, condition: str, generation: int, result: object) -> None:
        if generation != self._generations.get(condition) or not isinstance(result, CycleFit):
            return
        if not self.ready:
            return
        self.cycle_fits[condition] = result
        self._fill_cycles(self.cycle_fits.get("ec"), self.cycle_fits.get("eo"))
        if "ec" in self.cycle_fits and "eo" in self.cycle_fits:
            self.status.emit("Comparison ready.")

    def _cycle_failed(self, condition: str, generation: int, message: str) -> None:
        if generation != self._generations.get(condition):
            return
        detail = message.strip().splitlines()[-1] if message.strip() else "Cycle comparison failed."
        self.status.emit(detail)

    def _fill_cycles(self, eyes_closed: CycleFit | None, eyes_open: CycleFit | None) -> None:
        measures = [
            ("Period (ms)", "period_ms"),
            ("Rise–decay", "rise_decay"),
            ("Peak–trough", "peak_trough"),
            ("Voltage (µV)", "volt_amp"),
        ]
        for row, (label, field) in enumerate(measures):
            self.cycle_table.setItem(row, 0, _item(label))
            self.cycle_table.setItem(row, 1, _item(_num(None if eyes_closed is None else getattr(eyes_closed, field), 3)))
            self.cycle_table.setItem(row, 2, _item(_num(None if eyes_open is None else getattr(eyes_open, field), 3)))
        notes = []
        for name, fit in (("Eyes closed", eyes_closed), ("Eyes open", eyes_open)):
            if fit is not None and fit.burst_note:
                notes.append(f"{name}: {fit.burst_note}")
        self.cycle_note.setText(" ".join(notes))
        self.cycle_note.setVisible(bool(notes))

    def _empty(self, message: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#63748a", fontsize=12)
        self.canvas.draw_idle()


def _by_name(fit: SpectrumFit | None, name: str) -> ChannelFit | None:
    if fit is None:
        return None
    for item in fit.channels:
        if item.name == name:
            return item
    return None


def _plot_condition(ax, item: ChannelFit | None, color: str, label: str,
                    alpha: tuple[float, float]) -> None:
    if item is None or item.freqs is None or item.spectrum is None:
        return
    ax.plot(item.freqs, item.spectrum, color=color, lw=1.8, label=label)
    if item.aperiodic is not None:
        ax.plot(item.freqs, item.aperiodic, color=color, lw=1.2, ls="--", alpha=0.85)
    peak = alpha_peak(item, alpha)
    if peak is not None and item.aperiodic is not None:
        base = float(np.interp(peak[0], item.freqs, item.aperiodic))
        ax.plot(peak[0], base + peak[1], "o", color=color, ms=6)


def _num(value: float | None, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _item(text: str) -> QTableWidgetItem:
    cell = QTableWidgetItem(text)
    cell.setTextAlignment(Qt.AlignCenter)
    return cell
