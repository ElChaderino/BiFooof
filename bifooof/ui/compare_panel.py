"""Eyes-closed versus eyes-open comparison."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..analysis.compare import BAND_CHANGE, ChannelComparison, alpha_peak, compare_spectra
from ..analysis.cycles import CycleFit, fit_cycles
from ..analysis.prepare import R2_WEAK
from ..analysis.settings import FitSettings, band_label, band_phrase, matched_settings, reference_phrase
from ..analysis.specparam_fit import ChannelFit, SpectrumFit
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from ..plots import clear_figure, mark_spectrum, stamp_figure
from .canvas import MplCanvas, finish_figure, style_axes
from .headmap import draw_head
from .workers import CallWorker, launch, wait_for

_EC = "#1c4e78"
_EO = "#c2410c"


class ComparePanel(QWidget):
    channel_selected = Signal(str)
    status = Signal(str)
    updated = Signal()

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
        self.mode = QComboBox()
        self.mode.addItem("Eyes closed / Eyes open", "state")
        self.mode.addItem("Pre / Post", "pre")
        self.mode.addItem("Custom", "custom")
        self.mode.setToolTip(
            "Eyes closed / Eyes open is the usual pair. Pre / Post labels the same two files "
            "in the order of the open buttons. A difference between sessions is not a treatment effect."
        )
        self.custom_left = QLineEdit("A")
        self.custom_right = QLineEdit("B")
        self.custom_left.setFixedWidth(88)
        self.custom_right.setFixedWidth(88)
        self.custom_left.setVisible(False)
        self.custom_right.setVisible(False)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.custom_left.textChanged.connect(self._on_mode)
        self.custom_right.textChanged.connect(self._on_mode)
        pair = QHBoxLayout()
        pair_label = QLabel("Pair")
        pair_label.setObjectName("Muted")
        pair.addWidget(pair_label)
        pair.addWidget(self.mode)
        pair.addWidget(self.custom_left)
        pair.addWidget(self.custom_right)
        pair.addStretch(1)
        self.canvas = MplCanvas(width=8.0, height=6.4)
        self.canvas.setMinimumHeight(340)
        self.table = QTableWidget(0, 17)
        self._refresh_headers()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(480)
        self.table.setMaximumWidth(620)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_table)

        self.cycle_table = QTableWidget(8, 3)
        self.cycle_table.setHorizontalHeaderLabels(["Measure", "Eyes closed", "Eyes open"])
        self.cycle_table.verticalHeader().setVisible(False)
        self.cycle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cycle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cycle_table.setMaximumHeight(268)
        self.cycle_table.setMinimumWidth(480)
        self.cycle_table.setMaximumWidth(620)
        self.cycle_table.setAlternatingRowColors(True)
        self.cycle_note = QLabel("")
        self.cycle_note.setObjectName("Muted")
        self.cycle_note.setWordWrap(True)
        self.cycle_note.setVisible(False)

        plots = QVBoxLayout()
        plots.setSpacing(6)
        plots.addLayout(pair)
        plots.addWidget(self.caption)
        plots.addWidget(self.canvas, 1)
        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(self.cycle_table)
        side.addWidget(self.cycle_note)
        side.addWidget(self.table, 1)
        root = QHBoxLayout(self)
        root.setSpacing(10)
        root.addLayout(plots, 1)
        root.addLayout(side)
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
            self.rows = compare_spectra(
                self.fits["ec"], self.fits["eo"], self._alpha(), self.settings.peak_width_low,
            )
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
        self.rows = compare_spectra(fits["ec"], fits["eo"], self._alpha(), self.settings.peak_width_low)
        if channel and any(row.name == channel for row in self.rows):
            self.channel = channel
        elif self.rows and self.channel not in {row.name for row in self.rows}:
            self.channel = self.rows[0].name
        self._fill_table()
        self._draw()
        self._maybe_cycles()

    def shutdown(self) -> None:
        wait_for(self)

    def _names(self) -> tuple[str, str, str, str]:
        mode = self.mode.currentData()
        if mode == "pre":
            return "Pre", "Post", "Pre", "Post"
        if mode == "custom":
            left = self.custom_left.text().strip() or "A"
            right = self.custom_right.text().strip() or "B"
            return left, right, left, right
        return "Eyes closed", "Eyes open", "EC", "EO"

    def _on_mode(self, *_args) -> None:
        custom = self.mode.currentData() == "custom"
        self.custom_left.setVisible(custom)
        self.custom_right.setVisible(custom)
        self._refresh_headers()
        if not self.ready:
            return
        self._fill_table()
        self._draw()
        self._fill_cycles(self.cycle_fits.get("ec"), self.cycle_fits.get("eo"))

    def _refresh_headers(self) -> None:
        name = band_label(*self._alpha())
        left, right, short_left, short_right = self._names()
        self.table.setHorizontalHeaderLabels([
            "Channel", f"Offset {short_left}", f"Offset {short_right}",
            f"Exp {short_left}", f"Exp {short_right}", f"{short_right} − {short_left}",
            f"{name} {short_left}", f"{name} {short_right}", "Peak Δ", "Δ band %",
            f"CF {short_left}", f"CF {short_right}", "Δf",
            f"R² {short_left}", f"R² {short_right}",
            f"Nearest {short_left}", f"Nearest {short_right}",
        ])
        if hasattr(self, "cycle_table"):
            self.cycle_table.setHorizontalHeaderLabels(["Measure", left, right])
        header = self.table.horizontalHeaderItem(8)
        if header is not None:
            header.setToolTip(
                f"{left} fitted peak height minus {right.lower()} fitted peak height, "
                "above the aperiodic background, in log10 power. "
                "Each side picks its own tallest peak in the band. A missing peak stays blank. "
                "This is not the Welch band change."
            )
        for index, side in ((6, left), (7, right)):
            peak_header = self.table.horizontalHeaderItem(index)
            if peak_header is not None:
                peak_header.setToolTip(
                    f"{side} fitted peak height above the aperiodic background, in log10 power. "
                    "Not mean Welch power in the band."
                )
        band_header = self.table.horizontalHeaderItem(9)
        if band_header is not None:
            band_header.setToolTip(BAND_CHANGE)
        for index, tip in (
            (12, "Absolute difference between the two in-band center frequencies, in Hz."),
            (13, f"specparam R² for {left.lower()}. Below 0.6 the site is ringed on the head map."),
            (14, f"specparam R² for {right.lower()}. Below 0.6 the site is ringed on the head map."),
            (15, f"Closest {left.lower()} peak outside the band, and how far outside it sits. The band difference stays blank."),
            (16, f"Closest {right.lower()} peak outside the band, and how far outside it sits. The band difference stays blank."),
        ):
            header = self.table.horizontalHeaderItem(index)
            if header is not None:
                header.setToolTip(tip)

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
                _num(row.band_power_reactivity_percent, 1),
                _num(row.alpha_cf_ec, 2),
                _num(row.alpha_cf_eo, 2),
                _num(row.peak_center_difference_hz, 2),
                _num(row.r_squared_ec, 2),
                _num(row.r_squared_eo, 2),
                _nearest(row.nearest_cf_ec, row.nearest_distance_ec),
                _nearest(row.nearest_cf_eo, row.nearest_distance_eo),
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
        clear_figure(fig)
        finish_figure(fig)
        outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06)
        curves = outer[0, 0].subgridspec(4, 1, height_ratios=[1.25, 1.2, 1.1, 0.72], hspace=0.06)
        maps = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
        ax = fig.add_subplot(curves[:3, 0])
        ec_item = _by_name(self.fits.get("ec"), self.channel)
        eo_item = _by_name(self.fits.get("eo"), self.channel)
        fmin = min(item.fmin for item in self.fits.values())
        fmax = max(item.fmax for item in self.fits.values())
        mark_spectrum(ax, fmin, fmax)
        alpha = self._alpha()
        shown = band_phrase(*alpha)
        side_left, side_right, short_left, short_right = self._names()
        _plot_condition(ax, ec_item, _EC, side_left, alpha)
        _plot_condition(ax, eo_item, _EO, side_right, alpha)
        ax.set_xlabel("")
        ax.set_ylabel("log10 power")
        pair = "eyes closed vs eyes open" if self.mode.currentData() == "state" else f"{side_left} vs {side_right}"
        ax.set_title(
            f"{self.channel}   ·   {pair}   ·   {shown}",
            fontsize=10, loc="left",
        )
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax, grid=True)

        names = [row.name for row in self.rows]
        weak = [_weak(row.r_squared_ec) or _weak(row.r_squared_eo) for row in self.rows]
        draw_head(
            fig.add_subplot(maps[0, 0]), names,
            [row.exponent_delta for row in self.rows],
            f"Exponent  {fmin:g}–{fmax:.0f} Hz  {short_right} − {short_left}", "viridis", self.channel, weak,
        )
        draw_head(
            fig.add_subplot(maps[1, 0]), names,
            [row.alpha_reactivity for row in self.rows],
            f"{shown} peak height  {short_left} − {short_right}", "plasma", self.channel,
            [weak[index] or row.peak_at_floor_ec or row.peak_at_floor_eo for index, row in enumerate(self.rows)],
        )
        _paired_dots(fig.add_subplot(curves[3, 0]), self.rows, side_left, side_right)
        stamp_figure(fig)
        self.canvas.draw_idle()
        chosen = next((row for row in self.rows if row.name == self.channel), None)
        reactivity = None if chosen is None else chosen.alpha_reactivity
        percent = None if chosen is None else chosen.band_power_reactivity_percent
        percent_text = "blank" if percent is None else f"{percent:+.1f}%"
        name = band_label(*alpha)
        kind = name.lower() if name.isalpha() else name
        gap = ""
        if chosen is not None and chosen.peak_center_difference_hz is not None:
            gap = f" Centers are {chosen.peak_center_difference_hz:.2f} Hz apart."
        nearest = _nearest_sentence(chosen)
        quality = _quality_sentence(chosen)
        length = _length_sentence(self.recordings, self.fits, self.settings)
        if self.mode.currentData() == "pre":
            prefix = (
                "Pre is the file opened as eyes closed, and post is the file opened as eyes open. "
                "A difference between sessions is not a treatment effect. "
            )
        elif self.mode.currentData() == "custom":
            prefix = "A difference between two recordings is not a treatment effect. "
        else:
            prefix = ""
        self.caption.setText(
            f"{prefix}{self.channel}: {kind} peak height {short_left}−{short_right} {_num(reactivity, 3)} "
            "(above the aperiodic fit; a missing peak stays missing). "
            f"Δ band % {percent_text}: change in mean Welch power, ({short_left} − {short_right}) / {short_right}, "
            "not a share of the spectrum. "
            f"Slope fit over {fmin:g}–{fmax:.0f} Hz. {reference_phrase(self.settings.reference).capitalize()}."
            f"{gap}{nearest}{quality}{_voltage_sentence(self.recordings)}{length}"
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
            others = [
                item for key, item in self.recordings.items()
                if key != condition and item is not None
            ]
            worker = CallWorker(
                generation, fit_cycles,
                (recording, index, self.band, matched_settings(self.settings, recording, others)),
            )
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
        self.updated.emit()

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
            ("Burst cycles", "n_burst_cycles"),
            ("Burst episodes", "n_burst_episodes"),
            ("Burst time (s)", "burst_duration_s"),
            ("In bursts (% of accepted time)", "burst_fraction"),
        ]
        self.cycle_table.setRowCount(len(measures))
        for row, (label, field) in enumerate(measures):
            label_item = _item(label)
            if field == "burst_fraction":
                label_item.setToolTip(
                    "Percent of accepted recording time that falls inside detected bursts. "
                    "This is a share of time, not a change between recordings and not a share of spectral power."
                )
            self.cycle_table.setItem(row, 0, label_item)
            self.cycle_table.setItem(row, 1, _item(_cycle_value(eyes_closed, field)))
            self.cycle_table.setItem(row, 2, _item(_cycle_value(eyes_open, field)))
        notes = []
        left, right, _short_left, _short_right = self._names()
        for name, fit in ((left, eyes_closed), (right, eyes_open)):
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


def _paired_dots(ax, rows, left: str, right: str) -> None:
    """One point per side that has an in-band peak. A line needs both sides."""
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], [left, right])
    drawn = 0
    for row in rows:
        pair = []
        xpos = []
        if row.alpha_power_ec is not None and np.isfinite(row.alpha_power_ec):
            pair.append(float(row.alpha_power_ec))
            xpos.append(0)
        if row.alpha_power_eo is not None and np.isfinite(row.alpha_power_eo):
            pair.append(float(row.alpha_power_eo))
            xpos.append(1)
        if not pair:
            continue
        if len(pair) == 2:
            ax.plot(xpos, pair, color="#9aafc4", lw=0.7, zorder=1)
            drawn += 1
        color = _EC if xpos[0] == 0 else _EO
        ax.scatter([xpos[0]], [pair[0]], s=16, color=color, zorder=2, linewidths=0)
        if len(pair) == 2:
            ax.scatter([xpos[1]], [pair[1]], s=16, color=_EO, zorder=2, linewidths=0)
    ax.set_ylabel("Peak height")
    ax.set_title(
        f"One line per channel   ·   {drawn} with both peaks   ·   a missing peak is a dot",
        fontsize=9, loc="left",
    )
    style_axes(ax, grid=True)


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


def _weak(value: float | None) -> bool:
    return value is not None and value < R2_WEAK


def _nearest(center: float | None, distance: float | None) -> str:
    if center is None or distance is None:
        return "—"
    return f"{center:.2f} (+{distance:.2f})"


def _nearest_sentence(row: ChannelComparison | None) -> str:
    if row is None:
        return ""
    parts = []
    if row.nearest_cf_ec is not None and row.nearest_distance_ec is not None:
        parts.append(f"eyes closed nearest {row.nearest_cf_ec:.2f} Hz, {row.nearest_distance_ec:.2f} Hz outside")
    if row.nearest_cf_eo is not None and row.nearest_distance_eo is not None:
        parts.append(f"eyes open nearest {row.nearest_cf_eo:.2f} Hz, {row.nearest_distance_eo:.2f} Hz outside")
    if not parts:
        return ""
    return " " + "; ".join(parts) + "."


def _quality_sentence(row: ChannelComparison | None) -> str:
    if row is None:
        return ""
    notes = []
    if _weak(row.r_squared_ec) or _weak(row.r_squared_eo):
        notes.append("a weak fit is ringed")
    if row.peak_at_floor_ec or row.peak_at_floor_eo:
        notes.append("a peak is at the narrowest allowed width")
    if not notes:
        return ""
    return " " + "; ".join(notes).capitalize() + "."


def _length_sentence(recordings: dict, fits: dict, settings: FitSettings) -> str:
    if settings.t_length is not None or settings.t_start > 0.05:
        return ""
    durations = [recording.duration_s for recording in recordings.values()]
    if len(durations) < 2 or max(durations) - min(durations) <= 1.0:
        return ""
    ends = [fit.t_end - fit.t_start for fit in fits.values()]
    if not ends:
        return ""
    return (
        f" Files differ in length ({min(durations):.0f} s and {max(durations):.0f} s). "
        f"This comparison uses the first {min(ends):.0f} s of each."
    )


def _voltage_sentence(recordings: dict) -> str:
    warned = any(
        "microvolts" in note
        for recording in recordings.values()
        for note in recording.notes
    )
    if not warned:
        return ""
    return " Check voltage: the scale may not be microvolts, and the samples were not rescaled."


def _num(value: float | None, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _cycle_value(fit: CycleFit | None, field: str) -> str:
    if fit is None:
        return "—"
    value = getattr(fit, field)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if field == "burst_fraction":
        if value is None or not np.isfinite(value):
            return "—"
        return f"{100.0 * float(value):.1f}"
    return _num(value, 3)


def _item(text: str) -> QTableWidgetItem:
    cell = QTableWidgetItem(text)
    cell.setTextAlignment(Qt.AlignCenter)
    return cell
