"""BiFooof main window."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..analysis.compare import compare_spectra
from ..export import (
    comparison_rows, cycle_detail_rows, cycle_summary_row, fit_metadata, peak_rows,
    save_figure, spectrum_rows, write_csv, write_json,
)
from ..io.condition import CONDITION_LABELS, CONDITIONS, guess_condition
from ..io.marks import span_text, split_marked_conditions
from ..io.load import load_recording, needs_sfreq
from ..io.recording import Recording
from ..io.sniff import FILE_FILTER
from . import icons
from .compare_panel import ComparePanel
from .fit_bar import FitBar
from .cycle_panel import CyclePanel
from .spectrum_panel import SpectrumPanel, render_spectrum
from .theme import ACCENT, MUTED
from .trace_panel import TracePanel
from .workers import CallWorker, launch, wait_for


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"BiFooof  —  spectral studio  v{__version__}")
        self.resize(1360, 880)
        self.setMinimumSize(1120, 700)
        self.setWindowIcon(icons.app_icon())
        self.recordings: dict[str, Recording] = {}
        self.recording: Recording | None = None
        self.active = "ec"
        self._selected = 0
        self._directory = str(Path.home())
        self._load_generations: dict[str, int] = {}
        self._load_workers: dict[str, CallWorker] = {}

        header = QFrame()
        header.setObjectName("Header")
        logo = QLabel()
        logo.setPixmap(icons.pixmap("logo", ACCENT, 34, width=1.7))
        brand = QLabel("BiFooof")
        brand.setObjectName("Brand")
        sub = QLabel("SPECPARAM  ·  BYCYCLE")
        sub.setObjectName("BrandSub")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(brand)
        titles.addWidget(sub)

        self.view = QComboBox()
        self.view.setMinimumWidth(150)
        self.view.setToolTip("Which recording the traces, spectrum, and cycles are showing. Both keep running.")
        self.view.currentIndexChanged.connect(self._on_view_changed)
        viewing = QLabel("VIEWING")
        viewing.setObjectName("H2")
        self.file_label = QLabel("No recording open")
        self.file_label.setObjectName("Muted")

        self.open_ec = self._button(
            "Open eyes closed", "ec",
            "Load an eyes-closed recording. An eyes-open fit keeps running.",
        )
        self.open_eo = self._button(
            "Open eyes open", "eo",
            "Load an eyes-open recording. An eyes-closed fit keeps running.",
        )
        self.save_image_button = self._tool(
            "Save image", "image", self.save_image, "Save the current tab as a PNG.",
        )
        self.save_data_button = self._tool(
            "Save data", "table", self.save_data, "Save the current table as CSV.",
        )
        self.save_session_button = self._tool(
            "Save session", "layers", self._save_session_dialog,
            "Save both recordings, the comparison, and the plots into one folder.",
        )
        self.about_button = self._tool("", "info", self.show_about, "About BiFooof", ghost=True)
        self.about_button.setIcon(icons.icon("info", MUTED, 18))
        self.about_button.setFixedSize(36, 36)

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(logo)
        top.addLayout(titles)
        top.addSpacing(16)
        top.addWidget(viewing)
        top.addWidget(self.view)
        top.addSpacing(8)
        top.addWidget(self.file_label, 1)
        top.addWidget(self.about_button)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.open_ec)
        buttons.addWidget(self.open_eo)
        buttons.addStretch(1)
        buttons.addWidget(self.save_image_button)
        buttons.addWidget(self.save_data_button)
        buttons.addWidget(self.save_session_button)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 16, 10)
        header_layout.setSpacing(8)
        header_layout.addLayout(top)
        header_layout.addLayout(buttons)

        self.fit_bar = FitBar()
        self.fit_bar.apply_requested.connect(self._apply_fit)

        self.trace = TracePanel()
        self.spectrum = SpectrumPanel()
        self.cycles = CyclePanel()
        self.compare = ComparePanel()
        self.setStatusBar(QStatusBar())
        self.trace.channel_selected.connect(self._select)
        self.spectrum.channel_selected.connect(self._select)
        self.spectrum.fit_ready.connect(lambda _condition: self._sync_compare())
        self.spectrum.status.connect(self.statusBar().showMessage)
        self.cycles.status.connect(self.statusBar().showMessage)
        self.cycles.band.currentIndexChanged.connect(self._on_cycle_band)
        self.cycles.low.valueChanged.connect(self._on_cycle_band)
        self.cycles.high.valueChanged.connect(self._on_cycle_band)
        self.fit_bar.band_changed.connect(self._on_fit_band)
        self.compare.channel_selected.connect(self._select_name)
        self.compare.status.connect(self.statusBar().showMessage)

        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(16, 16))
        for widget, label in (
            (self.trace, "Traces"),
            (self.spectrum, "Spectrum"),
            (self.cycles, "Cycles"),
            (self.compare, "Compare"),
        ):
            self.tabs.addTab(widget, icons.icon(icons.TAB_ICONS[label], MUTED, 16), "  " + label)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.fit_bar)
        body = QVBoxLayout()
        body.setContentsMargins(12, 12, 12, 8)
        body.addWidget(self.tabs)
        layout.addLayout(body, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Open eyes-closed and eyes-open recordings. Both fits run together, then Compare opens."
        )

    def _button(self, label: str, condition: str, tip: str) -> QPushButton:
        button = QPushButton(f"  {label}")
        button.setIcon(icons.icon("folder", "#ffffff", 16))
        button.setIconSize(QSize(16, 16))
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False, cond=condition: self.open_recording(cond))
        return button

    def _tool(self, label: str, glyph: str, slot, tip: str, ghost: bool = False) -> QPushButton:
        button = QPushButton(f"  {label}" if label else "")
        button.setObjectName("Ghost" if ghost else "Secondary")
        button.setIcon(icons.icon(glyph, ACCENT, 16))
        button.setIconSize(QSize(16, 16))
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    def open_recording(self, condition: str) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, f"Open {CONDITION_LABELS[condition]}", self._directory, FILE_FILTER,
        )
        if not path:
            return
        self._directory = str(Path(path).parent)
        sfreq = None
        if needs_sfreq(path):
            value, accepted = QInputDialog.getDouble(
                self, "Sampling rate",
                "This file has no time column or sampling rate. Enter Hz:",
                256.0, 1.0, 100000.0, 2,
            )
            if not accepted:
                return
            sfreq = float(value)
        self._queue_load(path, sfreq, condition)

    def show_recording(self, recording: Recording, condition: str | None = None) -> None:
        """Show a recording that is already in memory. Used by the viewer and tests."""
        chosen = condition or guess_condition(recording.source_path) or self.active or "ec"
        self._apply(recording, chosen)

    def save_image(self) -> None:
        widget = self.tabs.currentWidget()
        if widget is self.trace and self.recording is None:
            self._tell("Open a recording before saving an image.")
            return
        if widget is self.spectrum and self.spectrum.fit is None:
            self._tell("The spectrum fit is not ready yet.")
            return
        if widget is self.cycles and self.cycles.fit is None:
            self._tell("The cycle fit is not ready yet.")
            return
        if widget is self.compare and not self.compare.ready:
            self._tell("Open both eyes-closed and eyes-open recordings before saving the comparison.")
            return
        path = self._choose_save("Save image", "png", "PNG image (*.png)")
        if not path:
            return
        widget.canvas.draw()
        destination = save_figure(widget.canvas.fig, path)
        self.statusBar().showMessage(f"Saved {destination.name}")

    def save_data(self) -> None:
        rows = self._current_rows()
        if not rows:
            self._tell("There is no table to save for this view yet.")
            return
        path = self._choose_save("Save data", "csv", "CSV table (*.csv)")
        if not path:
            return
        destination = write_csv(path, rows)
        self.statusBar().showMessage(f"Saved {destination.name}")

    def save_session(self, folder: str | Path) -> list[Path]:
        """Write the plots and tables that are ready into ``folder``."""
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        channel = self._channel_name()

        if self.recording is not None:
            self.trace.canvas.draw()
            written.append(save_figure(self.trace.canvas.fig, destination / "traces.png"))

        for key, fit in self.spectrum.fits.items():
            recording = self.recordings.get(key)
            index = 0
            if recording is not None and channel in recording.ch_names:
                index = recording.ch_names.index(channel)
            fig = Figure(figsize=(10, 7), dpi=120, facecolor="white")
            render_spectrum(fig, fit, index, CONDITION_LABELS.get(key, key))
            written.append(save_figure(fig, destination / f"{key}_spectrum.png"))
            source = "" if recording is None else recording.source_path
            written.append(write_csv(
                destination / f"{key}_spectrum.csv",
                spectrum_rows(fit, key, source),
            ))
            peaks = peak_rows(fit, key, source)
            if peaks:
                written.append(write_csv(destination / f"{key}_peaks.csv", peaks))
            if recording is not None:
                written.append(write_json(
                    destination / f"{key}_metadata.json",
                    fit_metadata(recording, self.spectrum.settings, fit, key),
                ))

        if self.cycles.fit is not None:
            self.cycles.canvas.draw()
            written.append(save_figure(self.cycles.canvas.fig, destination / "cycles.png"))
            details = cycle_detail_rows(self.cycles.fit, self.active)
            if details:
                written.append(write_csv(destination / f"{self.active}_cycles.csv", details))
            written.append(write_csv(
                destination / f"{self.active}_cycle_summary.csv",
                [cycle_summary_row(self.cycles.fit, self.active, self.recording.source_path if self.recording else "")],
            ))

        if self.compare.ready:
            self.compare.canvas.draw()
            written.append(save_figure(self.compare.canvas.fig, destination / "comparison.png"))
            band = self.fit_bar.settings().alpha_band()
            written.append(write_csv(
                destination / "comparison.csv",
                comparison_rows(compare_spectra(
                    self.spectrum.fits["ec"], self.spectrum.fits["eo"], band,
                ), band),
            ))
            summaries = []
            for key in CONDITIONS:
                fit = self.compare.cycle_fits.get(key)
                recording = self.recordings.get(key)
                if fit is not None:
                    summaries.append(cycle_summary_row(
                        fit, key, "" if recording is None else recording.source_path,
                    ))
                    details = cycle_detail_rows(fit, key)
                    if details:
                        written.append(write_csv(destination / f"{key}_compare_cycles.csv", details))
            if summaries:
                written.append(write_csv(destination / "cycle_comparison.csv", summaries))
        return written

    def _save_session_dialog(self) -> None:
        if not self.recordings:
            self._tell("Open a recording before saving a session.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save session", self._directory)
        if not folder:
            return
        self._directory = folder
        written = self.save_session(folder)
        self.statusBar().showMessage(f"Saved {len(written)} files to {folder}")

    def _current_rows(self) -> list[dict]:
        widget = self.tabs.currentWidget()
        if widget is self.spectrum and self.spectrum.fit is not None:
            source = "" if self.recording is None else self.recording.source_path
            return spectrum_rows(self.spectrum.fit, self.active, source)
        if widget is self.cycles and self.cycles.fit is not None:
            return cycle_detail_rows(self.cycles.fit, self.active)
        if widget is self.compare and self.compare.rows:
            return comparison_rows(self.compare.rows, self.compare.settings.alpha_band())
        if widget is self.trace and self.recording is not None:
            return [
                {
                    "condition": CONDITION_LABELS.get(self.active, self.active),
                    "source": self.recording.source_path,
                    "channel": name,
                    "sfreq_hz": self.recording.sfreq,
                    "duration_s": self.recording.duration_s,
                    "format": self.recording.source_format,
                }
                for name in self.recording.ch_names
            ]
        return []

    def _queue_load(self, path: str, sfreq: float | None, condition: str) -> None:
        self._load_generations[condition] = self._load_generations.get(condition, 0) + 1
        generation = self._load_generations[condition]
        button = self.open_ec if condition == "ec" else self.open_eo
        button.setEnabled(False)
        self.statusBar().showMessage(f"Reading {CONDITION_LABELS[condition].lower()}…")
        worker = CallWorker(generation, load_recording, (path, sfreq))
        worker.succeeded.connect(lambda gen, recording, cond=condition: self._loaded(cond, gen, recording))
        worker.failed.connect(lambda gen, message, cond=condition: self._load_failed(cond, gen, message))
        self._load_workers[condition] = worker
        launch(self, worker)

    def _loaded(self, condition: str, generation: int, recording: object) -> None:
        self._enable_open(condition)
        if generation != self._load_generations.get(condition) or not isinstance(recording, Recording):
            return
        self._present(recording, condition)

    def _present(self, recording: Recording, condition: str) -> None:
        """Open one recording, or both sides when its marks name them."""
        sides = split_marked_conditions(recording)
        if sides:
            order = [cond for cond in ("ec", "eo") if cond in sides and cond != condition]
            if condition in sides:
                order.append(condition)
            for cond in order:
                self._apply(sides[cond], cond)
            if len(sides) == 2:
                self.statusBar().showMessage(
                    "This file marks both eyes closed and eyes open, so both sides were opened."
                )
            elif condition not in sides:
                only = next(iter(sides))
                self.statusBar().showMessage(
                    f"This file's mark is {CONDITION_LABELS[only].lower()}, so it was opened on that side."
                )
            return
        hinted = guess_condition(recording.source_path)
        if hinted and hinted != condition:
            self.statusBar().showMessage(
                f"Opened as {CONDITION_LABELS[condition].lower()}. The filename looks like {CONDITION_LABELS[hinted].lower()}."
            )
        self._apply(recording, condition)

    def _load_failed(self, condition: str, generation: int, message: str) -> None:
        self._enable_open(condition)
        if generation != self._load_generations.get(condition):
            return
        detail = message.strip().splitlines()
        summary = detail[-1] if detail else "Could not open that file."
        self.statusBar().showMessage(summary)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Could not open recording")
        box.setText(summary)
        box.setDetailedText(message)
        box.exec()

    def _enable_open(self, condition: str) -> None:
        button = self.open_ec if condition == "ec" else self.open_eo
        button.setEnabled(True)

    def _apply(self, recording: Recording, condition: str) -> None:
        self.recordings[condition] = recording
        self.recording = recording
        self.active = condition
        self._selected = 0
        self._refresh_view_combo()
        self._refresh_label()
        note = "  ".join(recording.notes)
        if note:
            self.statusBar().showMessage(note)
        self.trace.condition_label = CONDITION_LABELS.get(condition, "")
        self.trace.set_recording(recording)
        self.spectrum.set_recording(recording, condition)
        self.cycles.set_recording(recording)
        self._sync_compare()

    def _on_view_changed(self, _index: int = 0) -> None:
        key = self.view.currentData()
        if not key or key == self.active or key not in self.recordings:
            return
        name = self._channel_name()
        recording = self.recordings[key]
        index = recording.ch_names.index(name) if name in recording.ch_names else 0
        self.active = key
        self.recording = recording
        self._selected = index
        self._refresh_label()
        self.trace.condition_label = CONDITION_LABELS.get(key, "")
        self.trace.set_recording(recording, index)
        self.spectrum._selected = index
        self.spectrum.show_condition(key)
        self.cycles.set_recording(recording, index)
        self._sync_compare()

    def _select(self, index: int) -> None:
        if self.recording is None:
            return
        if index == self._selected:
            self.trace.set_selected(index)
            return
        self._selected = index
        self.trace.set_selected(index)
        self.spectrum.set_selected(index)
        self.cycles.set_selected(index)
        self._sync_compare()

    def _select_name(self, name: str) -> None:
        if self.recording is None or name not in self.recording.ch_names:
            return
        self._select(self.recording.ch_names.index(name))

    def _apply_fit(self, settings) -> None:
        try:
            settings.validate()
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return
        self.spectrum.apply_settings(settings)
        self.cycles.apply_settings(settings)
        self.compare.apply_settings(settings)
        if settings.t_length is None and settings.t_start <= 0:
            window = "the whole recording"
        else:
            end = "the end" if settings.t_length is None else f"{settings.t_start + settings.t_length:.0f} s"
            window = f"{settings.t_start:.0f} s to {end}"
        self.statusBar().showMessage(
            f"Refitting {window}, {settings.fmin:g}–{settings.fmax:.0f} Hz, {settings.aperiodic_mode} slope."
        )

    def _on_fit_band(self, low: float, high: float) -> None:
        self.cycles.set_band(float(low), float(high), named=self.fit_bar.band.currentData() is not None)
        self._sync_compare()

    def _on_cycle_band(self, *_args) -> None:
        low, high = self.cycles.current_band()
        self.fit_bar.set_band(low, high, named=self.cycles.band.currentData() is not None)
        self._sync_compare()

    def _sync_compare(self) -> None:
        band = self.fit_bar.current_band()
        self.compare.set_pair(self.recordings, self.spectrum.fits, self._channel_name(), band)

    def _channel_name(self) -> str:
        if self.recording is None or not self.recording.ch_names:
            return ""
        index = min(max(self._selected, 0), len(self.recording.ch_names) - 1)
        return self.recording.ch_names[index]

    def _refresh_view_combo(self) -> None:
        self.view.blockSignals(True)
        self.view.clear()
        for key in CONDITIONS:
            if key in self.recordings:
                self.view.addItem(CONDITION_LABELS[key], key)
        index = self.view.findData(self.active)
        if index >= 0:
            self.view.setCurrentIndex(index)
        self.view.blockSignals(False)

    def _refresh_label(self) -> None:
        parts = []
        tips = []
        for key in CONDITIONS:
            recording = self.recordings.get(key)
            if recording is None:
                continue
            name = Path(recording.source_path).name
            span = span_text(recording.notes)
            shown = f"{CONDITION_LABELS[key]}: {name}"
            if span:
                shown = f"{shown} ({span})"
            parts.append(shown)
            detail = " ".join(recording.notes)
            tips.append(f"{shown}. {detail}".strip())
        text = "    ".join(parts) if parts else "No recording open"
        self.file_label.setText(text)
        self.file_label.setToolTip("\n".join(tips) if tips else text)

    def _tell(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.information(self, "BiFooof", message)

    def _choose_save(self, title: str, suffix: str, filt: str) -> str:
        suggested = str(Path(self._directory) / self._suggest(suffix))
        path, _selected = QFileDialog.getSaveFileName(self, title, suggested, filt)
        if path:
            self._directory = str(Path(path).parent)
        return path

    def _suggest(self, suffix: str) -> str:
        widget = self.tabs.currentWidget()
        if widget is self.compare:
            stem = "comparison"
        elif widget is self.spectrum:
            stem = f"{self.active}_spectrum"
        elif widget is self.cycles:
            stem = f"{self.active}_cycles"
        else:
            stem = f"{self.active}_traces"
        return f"{stem}.{suffix}"

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About BiFooof",
            f"<b>BiFooof {__version__}</b><br><br>"
            "Spectral parameterization with specparam, and cycle-by-cycle shape with bycycle.<br><br>"
            "Open an eyes-closed recording and an eyes-open recording. Both fits keep running. "
            "Compare overlays the two spectra, the exponent change, and reactivity "
            "in the selected band (eyes closed minus eyes open). Alpha 8–12 Hz is the default. "
            "Delta, theta, beta, gamma, and a custom range use the same views. "
            "A WinEEG file whose marks name eyes closed and eyes open opens both sides. "
            "When cycles are found and none form a burst, the cycle page names the cutoff that stopped the run.<br><br>"
            "Save image writes the current view. Save data writes the current table. "
            "Save session writes both conditions and the comparison into one folder.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        wait_for(self)
        self.spectrum.shutdown()
        self.cycles.shutdown()
        self.compare.shutdown()
        event.accept()
