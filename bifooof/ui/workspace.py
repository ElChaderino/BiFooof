"""Project and review orchestration shared by the desktop's feature panels."""
from copy import deepcopy
from dataclasses import asdict, replace
import base64
import csv
from io import BytesIO
import json
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QInputDialog,
    QLineEdit, QDoubleSpinBox, QPlainTextEdit, QComboBox, QCheckBox, QVBoxLayout, QTextBrowser, QMessageBox)

from ..analysis.insights import quality_rows, comparison_checks, epoch_rows, regional_rows
from ..analysis.quality import QualitySpan
from ..analysis.settings import window_bounds
from ..events import event_table
from ..project import (ReviewState, atomic_json, read_json, state_directory, make_project,
    read_project, load_project_sources, profile, read_profile, settings_from, source_files)
from ..report import MEASURES, report_html, write_report
from .review_panel import ReviewPanel
from .workers import CallWorker, launch


TIME_CONTROLS = ("mode", "spec_window", "spec_overlap", "slide_window", "slide_overlap", "lock_color", "color_low", "color_high",
    "top_trace", "map_which", "react_pre", "react_post", "bursts", "every", "align", "normalized",
    "phase_low", "phase_high", "amp_low", "amp_high", "comod", "event_direction")
TRACE_CONTROLS = ("window", "gain", "display_band", "notch", "montage")


def control_values(panel, names):
    result = {}
    for name in names:
        widget = getattr(panel, name, None)
        if isinstance(widget, QComboBox):
            result[name] = {"index": widget.currentIndex(), "data": widget.currentData(), "text": widget.currentText()}
        elif isinstance(widget, QCheckBox):
            result[name] = widget.isChecked()
        elif widget is not None and hasattr(widget, "value"):
            result[name] = widget.value()
    return result


def restore_controls(panel, values):
    for name, value in values.items():
        widget = getattr(panel, name, None)
        if widget is None:
            continue
        widget.blockSignals(True)
        try:
            if isinstance(widget, QComboBox):
                index = widget.findData(value.get("data")) if value.get("data") is not None else widget.findText(value.get("text", ""))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif hasattr(widget, "setValue"):
                widget.setValue(value)
        finally:
            widget.blockSignals(False)


class WorkspaceController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.state = ReviewState()
        self.project_path = None
        self.dirty = False
        self._restoring = False
        self._profile = None
        self._report_options = None
        self._project_generation = 0
        self._preferences_path = state_directory() / "preferences.json"
        self._recovery_path = state_directory() / "recovery.bifooof.json"
        try:
            self.preferences = read_json(self._preferences_path)
        except (OSError, ValueError):
            self.preferences = {}
        self.import_policy = self.preferences.get("import_policy", "standard")
        self.review = ReviewPanel()
        from .study_panel import StudyPanel
        self.study = StudyPanel(self)
        window.tabs.addTab(self.review, "Review")
        window.tabs.addTab(self.study, "Study")
        self.review.action_requested.connect(self.dispatch)
        self.review.inspect_requested.connect(self.inspect)
        for widget in (self.review.notes, self.review.alias, self.review.date):
            widget.textChanged.connect(self.note_changed)
        window.spectrum.fit_ready.connect(self.changed)
        window.cycles.fit_ready.connect(self.changed)
        window.trace.interval_selected.connect(self.selected_interval)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.timeout.connect(self.autosave)
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(500)
        self._note_timer.timeout.connect(self.flush_notes)
        self._install_menu()
        self.review.restore_notes(self.state)
        directory = self.preferences.get("directory")
        if directory and Path(directory).is_dir():
            window._directory = directory
        geometry = self.preferences.get("geometry")
        if geometry:
            window.restoreGeometry(bytes.fromhex(geometry))
        self.refresh()
        self._ui_signature = self._signature()

    def _signature(self):
        return repr((self.window.fit_bar.settings().key(), self.ui_state()))

    def check_changes(self):
        if not self._restoring:
            signature = self._signature()
            if signature != getattr(self, "_ui_signature", signature):
                self._ui_signature = signature
                self.changed()

    def _install_menu(self):
        menu = self.window.menuBar()
        for title, entries in (
            ("File", (("Open project…", "open_project", "Ctrl+P"), ("Save project", "save_project", "Ctrl+S"),
                ("Save project as…", "save_project_as", "Ctrl+Alt+S"), ("Recent projects…", "recent_project", ""),
                ("Recover autosave", "recover", ""), ("Export results…", "export_results", "Ctrl+Shift+E"),
                ("Report and results…", "report", "Ctrl+Alt+P"), ("Import channel selection…", "channel_policy", ""))),
            ("Review", (("Add bookmark…", "bookmark", "Ctrl+B"), ("Undo review", "undo", "Ctrl+Alt+Z"),
                ("Redo review", "redo", "Ctrl+Alt+Y"), ("Previous bookmark", "previous_bookmark", "Ctrl+["),
                ("Next bookmark", "next_bookmark", "Ctrl+]"))),
            ("Profiles", (("Save current profile…", "save_profile", ""), ("Apply saved profile…", "choose_profile", ""),
                ("Import profile…", "import_profile", ""), ("Export current profile…", "export_profile", ""))),
            ("Tools", (("Batch queue…", "batch", ""), ("Cancel background work", "cancel", "Esc"),
                ("Clear analysis caches", "clear_cache", ""))),
            ("Help", (("Measurement guide", "help", "F1"), ("Open synthetic demonstration", "demo", ""))),
        ):
            section = menu.addMenu(title)
            for label, name, shortcut in entries:
                action = QAction(label, self)
                if shortcut:
                    action.setShortcut(shortcut)
                action.triggered.connect(lambda _checked=False, chosen=name: self.dispatch(chosen))
                section.addAction(action)
        # Ctrl+S now saves a project; output export keeps its own explicit action.
        for shortcut in self.window._shortcuts:
            if shortcut.key().toString() == "Ctrl+S":
                shortcut.setEnabled(False)
        self.window.save_session_button.setText("  Export results")
        self.window.save_session_button.setToolTip("Export completed plots, tables, metadata, and review sidecars. Ctrl+Shift+E")

    def dispatch(self, name):
        try:
            getattr(self, name)()
        except Exception:
            import traceback
            self.window._show_error("Could not complete that action", traceback.format_exc())

    def changed(self, *_args):
        if self._restoring:
            return
        self.dirty = True
        self._autosave_timer.start()
        self._refresh_timer.start(100)
        self.title()

    def title(self):
        from .. import __version__
        name = "Unsaved project" if self.project_path is None else self.project_path.name
        self.window.setWindowTitle(f"BiFooof · {name}{' *' if self.dirty else ''} · v{__version__}")

    def recording_changed(self):
        if self._restoring:
            return
        self.changed()
        self.refresh()

    def note_changed(self):
        self._note_timer.start()
        self.changed()

    def flush_notes(self):
        self._note_timer.stop()
        self.state.change(notes=self.review.notes.toPlainText(), alias=self.review.alias.text(), date=self.review.date.text())

    def ui_state(self):
        window = self.window
        return {"active": window.active, "channel": window._channel_name(), "tab": window.tabs.currentIndex(),
            "trace": control_values(window.trace, TRACE_CONTROLS), "time": control_values(window.time, TIME_CONTROLS),
            "time_channel_controls": control_values(window.time, ("other", "amp_channel", "react_channel")),
            "time_picks": [window.time.picks.item(i).text() for i in range(window.time.picks.count()) if window.time.picks.item(i).checkState()],
            "cycles_time": window.cycles._t0, "trace_selection": window.trace.selection,
            "trace_time": window.trace._t0, "hidden_channels": [window.trace.channels.item(i).text() for i in range(window.trace.channels.count()) if not window.trace.channels.item(i).checkState()],
            "compare_mode": window.compare.mode.currentData(), "compare_left": window.compare.custom_left.text(),
            "compare_right": window.compare.custom_right.text(), "profile": deepcopy(self._profile)}

    def session_summaries(self):
        from ..analysis.compare import alpha_peak
        result = []
        cycles = dict(self.window.compare.cycle_fits)
        if self.window.cycles.fit is not None:
            cycles[self.window.active] = self.window.cycles.fit
        for condition, fit in self.window.spectrum.fits.items():
            for item in fit.channels:
                peak = alpha_peak(item, (fit.settings or self.window.spectrum.settings).alpha_band())
                cycle = cycles.get(condition)
                cycle = cycle if cycle is not None and cycle.channel == item.name else None
                result.append({"condition": condition, "channel": item.name, "exponent": item.exponent,
                    "peak_hz": None if peak is None else peak[0], "peak_height": None if peak is None else peak[1],
                    "r_squared": item.r_squared, "accepted_s": fit.provenance.get("channels", {}).get(item.name, {}).get("accepted_duration_s"),
                    "burst_fraction": None if cycle is None else cycle.burst_fraction, "burst_episodes": None if cycle is None else cycle.n_burst_episodes,
                    "settings": asdict(fit.settings or self.window.spectrum.settings), "reference": fit.provenance.get("reference_applied"),
                    "reference_contributors": fit.provenance.get("reference_contributors", []),
                    "stored_reference": self.window.recordings[condition].import_info.get("stored_reference", "Unknown"),
                    "profile": self._profile["name"] if self._profile else "Custom",
                    "sfreq": self.window.recordings[condition].sfreq})
        return result

    def snapshot(self):
        self.flush_notes()
        return make_project(self.window.sources, replace(self.window.spectrum.settings), self.window.fit_bar.settings(),
            self.state, self.ui_state(), summaries=self.session_summaries())

    def save_project(self):
        if not self.window.sources:
            self.window._tell("Open recordings before saving a project.")
            return
        if self.project_path is None:
            return self.save_project_as()
        atomic_json(self.project_path, self.snapshot())
        self.dirty = False
        self._remember_project(self.project_path)
        self.title()
        self.window.statusBar().showMessage(f"Saved project {self.project_path.name}")

    def save_project_as(self):
        path, _selected = QFileDialog.getSaveFileName(self.window, "Save project", str(Path(self.window._directory) / "review.bifooof.json"), "BiFooof project (*.bifooof.json)")
        if path:
            previous = self.project_path
            self.project_path = Path(path)
            try:
                self.save_project()
            except Exception:
                self.project_path = previous
                raise

    def autosave(self):
        if not self.dirty or not self.window.sources:
            return
        try:
            atomic_json(self._recovery_path, self.snapshot())
        except (OSError, ValueError) as exc:
            self.window.statusBar().showMessage(f"Recovery copy could not be saved: {exc}")

    def _remember_project(self, path):
        recent = [str(Path(path).resolve()), *self.preferences.get("recent_projects", [])]
        self.preferences["recent_projects"] = list(dict.fromkeys(recent))[:15]
        atomic_json(self._preferences_path, self.preferences)

    def open_project(self, path=None):
        if path is None:
            path, _selected = QFileDialog.getOpenFileName(self.window, "Open project", self.window._directory, "BiFooof project (*.json)")
        if not path:
            return
        self.autosave()
        value = read_project(path)
        replacements = {}
        for condition, source in value["sources"].items():
            missing = not Path(source["path"]).is_file()
            if not missing:
                missing = any(not file.is_file() for file in source_files(source["path"]).values())
            if missing:
                replacement, _selected = QFileDialog.getOpenFileName(self.window, f"Relink {condition.upper()}: {Path(source['path']).name}", self.window._directory)
                if not replacement:
                    return
                replacements[condition] = replacement
        self._project_generation += 1
        worker = CallWorker(self._project_generation, load_project_sources, (value, replacements))
        worker.succeeded.connect(lambda generation, result: self.restore_project(value, result, Path(path)) if generation == self._project_generation else None)
        worker.failed.connect(lambda generation, error: self.window._show_error("Could not reopen project", error) if generation == self._project_generation and not error.startswith("Cancelled") else None)
        launch(self, worker, key="project")
        self.window.statusBar().showMessage("Verifying project sources…")

    def restore_project(self, project, loaded, path=None):
        window = self.window
        if window._closing:
            return
        self.cancel()
        for owner in (window, window.spectrum, window.compare):
            generations = owner._load_generations if owner is window else owner._generations
            for condition in list(generations):
                generations[condition] += 1
        self._restoring = True
        try:
            sources, continuous = loaded
            self.state = ReviewState()
            self.state.restore(project.get("review", {}))
            self.state.restore_history(project.get("review_history", {}))
            # Event edits follow a verified relocated source, including undo history.
            moved = {project["sources"][condition]["path"]: source.source_path for condition, source in sources.items()}
            for review in [self.state.value, *self.state._undo, *self.state._redo]:
                review["events"] = {moved.get(key, key): events for key, events in review.get("events", {}).items()}
            self.review.restore_notes(self.state)
            applied = settings_from(project["applied"])
            window.sources, window.recordings = dict(sources), {}
            window.spectrum.recordings, window.spectrum.fits = {}, {}
            window.spectrum.settings = replace(applied)
            window.time.settings = replace(applied)
            window.compare.settings = replace(applied)
            window.continuous = continuous
            window.time.excluded = {condition: [QualitySpan(**item) for item in items] for condition, items in self.state.value["exclusions"].items()}
            window.time.review_events = deepcopy(self.state.value["events"])
            for condition, source in sources.items():
                window._apply(self.state.apply(source, condition), condition, remember=False)
            ui = project.get("ui", {})
            active = ui.get("active", window.active)
            index = window.view.findData(active)
            if index >= 0:
                window.view.setCurrentIndex(index)
            window._select_name(ui.get("channel", window._channel_name()))
            restore_controls(window.trace, ui.get("trace", {}))
            window.trace._win_s = float(window.trace.window.currentData())
            window.trace._gain = float(window.trace.gain.currentData())
            window.trace.show_time(float(ui.get("trace_time", 0)) + window.trace._win_s / 2)
            for index in range(window.trace.channels.count()):
                item = window.trace.channels.item(index)
                item.setCheckState(Qt.Unchecked if item.text() in ui.get("hidden_channels", []) else Qt.Checked)
            restore_controls(window.time, ui.get("time", {}))
            restore_controls(window.time, ui.get("time_channel_controls", {}))
            window.time.picks.blockSignals(True)
            for index in range(window.time.picks.count()):
                item = window.time.picks.item(index)
                item.setCheckState(Qt.Checked if item.text() in ui.get("time_picks", []) else Qt.Unchecked)
            window.time.picks.blockSignals(False)
            window.time._color_limits = (window.time.color_low.value(), window.time.color_high.value()) if window.time.lock_color.isChecked() else None
            window.trace.selection = ui.get("trace_selection")
            window.cycles._t0 = float(ui.get("cycles_time", 0))
            window.time._kick()
            mode_index = window.compare.mode.findData(ui.get("compare_mode", "state"))
            window.compare.mode.setCurrentIndex(max(0, mode_index))
            window.compare.custom_left.setText(ui.get("compare_left", ""))
            window.compare.custom_right.setText(ui.get("compare_right", ""))
            window.fit_bar.set_settings(settings_from(project.get("staged", project["applied"])))
            window.fit_bar.mark_applied(applied)
            self._profile = ui.get("profile")
            self.project_path = path
            self.dirty = False
            window.tabs.setCurrentIndex(min(int(ui.get("tab", 0)), window.tabs.count() - 1))
            self.title()
            if path:
                self._remember_project(path)
        finally:
            self._restoring = False
        self.refresh()
        self._ui_signature = self._signature()

    def recent_project(self):
        paths = self.preferences.get("recent_projects", [])
        if not paths:
            self.window._tell("Saved projects will appear here.")
            return
        selected, accepted = QInputDialog.getItem(self.window, "Recent projects", "Project", paths, editable=False)
        if accepted:
            self.open_project(selected)

    def recover(self):
        if not self._recovery_path.exists():
            self.window._tell("There is no recovery copy yet.")
            return
        self.open_project(str(self._recovery_path))

    def selected_interval(self, start, stop):
        self.window.time._inspection = ((start + stop) / 2, stop - start)
        self.window.time._draw()

    def record_exclusions(self):
        self.state.change(exclusions={condition: [asdict(item) for item in items] for condition, items in self.window.time.excluded.items()})
        self.changed()

    def apply_review(self):
        window = self.window
        window.time.excluded = {condition: [QualitySpan(**item) for item in items] for condition, items in self.state.value["exclusions"].items()}
        window.time.review_events = deepcopy(self.state.value["events"])
        for condition, source in list(window.sources.items()):
            window.recordings[condition] = self.state.apply(source, condition)
        active = window.active
        for condition, recording in list(window.recordings.items()):
            if condition != active:
                window.spectrum.set_recording(recording, condition)
        if active in window.recordings:
            window._apply(window.recordings[active], active, remember=False, keep_channel=True)
        self.review.restore_notes(self.state)
        self.changed()
        self.refresh()

    def undo(self):
        self.flush_notes()
        self.state.undo()
        self.apply_review()

    def redo(self):
        self.state.redo()
        self.apply_review()

    def _selected_quality(self):
        index = self.review.quality.currentRow()
        return self.review.quality_rows[index] if 0 <= index < len(self.review.quality_rows) else None

    def inspect_channel(self):
        item = self._selected_quality()
        if item:
            self.inspect(item["condition"], item["channel"], -1)

    def inspect(self, condition=None, channel=None, moment=-1):
        if condition is None:
            item = self._selected_quality()
            if item is None:
                return
            condition, channel = item["condition"], item["channel"]
        index = self.window.view.findData(condition)
        if index >= 0:
            self.window.view.setCurrentIndex(index)
        self.window._select_name(channel or "")
        if moment >= 0:
            self.window._inspect(moment, 4)
        self.window.tabs.setCurrentWidget(self.window.trace if moment >= 0 else self.window.spectrum)

    def reviewed(self):
        item = self._selected_quality()
        if item:
            key = f"{item['condition']}:{item['channel']}"
            self.state.change(reviewed=list(dict.fromkeys([*self.state.value["reviewed"], key])))
            self.changed()

    def exclude_channel(self):
        self._channel_exclusion(True)

    def restore_channel(self):
        self._channel_exclusion(False)

    def _channel_exclusion(self, excluded):
        item = self._selected_quality()
        if item:
            choices = deepcopy(self.state.value["bad_channels"])
            names = choices.setdefault(item["condition"], [])
            if excluded and item["channel"] not in names:
                names.append(item["channel"])
            if not excluded and item["channel"] in names:
                names.remove(item["channel"])
            self.state.change(bad_channels=choices)
            self.apply_review()

    def bookmark(self, existing=None):
        if self.window.recording is None:
            return
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Review bookmark")
        form = QFormLayout(dialog)
        condition = QComboBox()
        condition.addItems(list(self.window.recordings))
        condition.setCurrentText(self.window.active if existing is None else existing["condition"])
        channel = QLineEdit(self.window._channel_name() if existing is None else existing.get("channel", "*"))
        start, duration = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (start, duration):
            widget.setRange(0, 1e8)
            widget.setDecimals(3)
            widget.setSuffix(" s")
        selection = self.window.trace.selection
        start.setValue(existing["time_s"] if existing else selection[0] if selection else self.window.trace._t0)
        duration.setValue(existing.get("duration_s", 0) if existing else selection[1] - selection[0] if selection else 0)
        label = QLineEdit(existing.get("label", "") if existing else "")
        note = QPlainTextEdit(existing.get("note", "") if existing else "")
        for title, widget in (("Condition", condition), ("Channel (* for all)", channel), ("Start", start), ("Duration", duration), ("Label", label), ("Observation", note)):
            form.addRow(title, widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if not dialog.exec():
            return
        source = self.window.recordings[condition.currentText()]
        if start.value() >= source.duration_s or start.value() + duration.value() > source.duration_s or channel.text().strip() not in ["*", "", *source.ch_names]:
            raise ValueError("Choose an interval and channel present in that recording.")
        entry = {"condition": condition.currentText(), "channel": channel.text().strip() or "*", "time_s": start.value(),
                 "duration_s": duration.value(), "label": label.text().strip() or "Bookmark", "note": note.toPlainText()}
        bookmarks = deepcopy(self.state.value["bookmarks"])
        if existing is not None:
            bookmarks[bookmarks.index(existing)] = entry
        else:
            bookmarks.append(entry)
        bookmarks.sort(key=lambda item: (item["condition"], item["time_s"]))
        self.state.change(bookmarks=bookmarks)
        self.changed()

    def _selected_bookmark(self):
        index = self.review.bookmarks.currentRow()
        rows = self.state.value["bookmarks"]
        return rows[index] if 0 <= index < len(rows) else None

    def edit_bookmark(self):
        item = self._selected_bookmark()
        if item:
            self.bookmark(item)

    def delete_bookmark(self):
        item = self._selected_bookmark()
        if item:
            self.state.change(bookmarks=[row for row in self.state.value["bookmarks"] if row != item])
            self.changed()

    def exclude_bookmark(self):
        item = self._selected_bookmark()
        if item and item["duration_s"] > 0:
            exclusions = deepcopy(self.state.value["exclusions"])
            exclusions.setdefault(item["condition"], []).append(asdict(QualitySpan(item["channel"], item["time_s"], item["time_s"] + item["duration_s"], "user", "confirmed", item["label"])))
            self.state.change(exclusions=exclusions)
            self.apply_review()

    def next_bookmark(self):
        self._move_bookmark(1)

    def previous_bookmark(self):
        self._move_bookmark(-1)

    def _move_bookmark(self, direction):
        count = len(self.state.value["bookmarks"])
        if count:
            index = (self.review.bookmarks.currentRow() + direction) % count
            self.review.bookmarks.selectRow(index)
            self.review.inspect_bookmark(index)

    def refresh(self):
        self.review.refresh(self.window.recordings, self.window.spectrum.fits, self.state)
        self.study.refresh()
        if self._profile:
            modified = self.window.fit_bar.settings().key() != settings_from(self._profile["settings"]).key() or control_values(self.window.time, TIME_CONTROLS) != self._profile.get("time_controls", {})
            self.review.profile_label.setText(f"Processing profile: {self._profile['name']}{' (modified)' if modified else ''}")
        self.window.trace.fit_window = None if self.window.recording is None else window_bounds(self.window.recording, self.window._cycle_settings(self.window.spectrum.settings))
        self.window.trace.overlays = [item for item in self.state.value["bookmarks"] if item["condition"] == self.window.active]
        self.window.trace.overlays += [{**item, "excluded": True} for item in self.state.value["exclusions"].get(self.window.active, [])]

    def save_profile(self):
        name, accepted = QInputDialog.getText(self.window, "Save processing profile", "Name")
        if accepted and name.strip():
            from uuid import uuid4
            self._profile = profile(self.window.fit_bar.settings(), control_values(self.window.time, TIME_CONTROLS), name.strip())
            atomic_json(state_directory() / "profiles" / f"{uuid4().hex}.json", self._profile)
            self.changed()

    def choose_profile(self):
        files = sorted((state_directory() / "profiles").glob("*.json"))
        profiles = [read_profile(path) for path in files]
        if not profiles:
            self.window._tell("Save or import a profile first.")
            return
        names = [f"{index + 1}. {item['name']}" for index, item in enumerate(profiles)]
        name, accepted = QInputDialog.getItem(self.window, "Processing profiles", "Profile", names, editable=False)
        if accepted:
            self.apply_profile(profiles[names.index(name)])

    def apply_profile(self, selected, preview=True):
        settings = settings_from(selected["settings"])
        changes = [f"{key}: {getattr(self.window.spectrum.settings, key)} → {value}" for key, value in asdict(settings).items() if getattr(self.window.spectrum.settings, key) != value]
        if preview and QMessageBox.question(self.window, f"Apply {selected['name']}", "\n".join(changes) or "Settings match. Apply this profile?") != QMessageBox.Yes:
            return
        self.window.fit_bar.set_settings(settings)
        restore_controls(self.window.time, selected.get("time_controls", {}))
        self.window.cycles.set_band(*settings.alpha_band(), named=True)
        self.window._apply_fit(settings)
        self._profile = deepcopy(selected)
        self.changed()

    def import_profile(self):
        path, _selected = QFileDialog.getOpenFileName(self.window, "Import profile", self.window._directory, "JSON profile (*.json)")
        if path:
            selected = read_profile(path)
            from uuid import uuid4
            atomic_json(state_directory() / "profiles" / f"{uuid4().hex}.json", selected)
            self.apply_profile(selected)

    def export_profile(self):
        path, _selected = QFileDialog.getSaveFileName(self.window, "Export profile", str(Path(self.window._directory) / "processing-profile.json"), "JSON profile (*.json)")
        if path:
            atomic_json(path, profile(self.window.fit_bar.settings(), control_values(self.window.time, TIME_CONTROLS), self._profile["name"] if self._profile else "Custom"))

    def export_results(self):
        self.window._save_session_dialog()

    def export_time_data(self):
        from ..export import write_csv_stream
        from ..time_export import time_rows
        if not self.window.time.data_ready:
            self.window._tell("There is no completed table for this Time view.")
            return
        path = self.window._choose_save("Save Time data", "csv", "CSV table (*.csv)")
        if not path:
            return
        snapshot = self.window.time.export_snapshot()
        worker = CallWorker(1, lambda: write_csv_stream(path, time_rows(snapshot)))
        worker.succeeded.connect(lambda _generation, destination: self.window.statusBar().showMessage(f"Saved Time snapshot: {destination.name}"))
        worker.failed.connect(lambda _generation, message: self.window._show_error("Could not save Time data", message))
        launch(self, worker, key="export")

    def export_session_async(self, folder):
        import shutil
        import tempfile
        from ..export import write_csv_stream
        from ..session_export import save_bundle
        from ..time_export import time_rows
        capture = tempfile.TemporaryDirectory(prefix="bifooof-snapshot-")
        self.window._skip_time_export = True
        try:
            self.window._write_session(capture.name)
            self.window.time.canvas.fig.savefig(Path(capture.name) / "time.png", dpi=150, bbox_inches="tight")
            snapshot = self.window.time.export_snapshot()
        except Exception:
            capture.cleanup()
            raise
        finally:
            self.window._skip_time_export = False
        def save():
            try:
                def writer(stage):
                    for file in Path(capture.name).iterdir():
                        shutil.copy2(file, stage / file.name)
                    write_csv_stream(stage / "time.csv", time_rows(snapshot))
                return save_bundle(folder, writer)
            finally:
                capture.cleanup()
        worker = CallWorker(1, save)
        worker.succeeded.connect(lambda _generation, paths: self.window.statusBar().showMessage(f"Exported {len(paths)} files from one analysis snapshot to {folder}"))
        worker.failed.connect(lambda _generation, message: self.window._show_error("Could not export results", message))
        launch(self, worker, key="export")

    def channel_policy(self):
        choices = ["Standard 19 EEG channels", "All non-marker channels (inspect types)"]
        selected, accepted = QInputDialog.getItem(self.window, "Future imports", "Channel selection", choices, 0 if self.import_policy == "standard" else 1, False)
        if accepted:
            self.import_policy = "standard" if selected == choices[0] else "all"
            self.preferences["import_policy"] = self.import_policy
            atomic_json(self._preferences_path, self.preferences)

    def cancel(self):
        for owner in self.owners():
            for worker in getattr(owner, "_live_workers", []):
                worker.cancel()
        self.window.statusBar().showMessage("Cancellation requested. Active library calls finish at their next safe stopping point.")

    def clear_cache(self):
        self.window.time._cache.clear()
        self.window.statusBar().showMessage("Unused Time results cleared; current results remain available.")

    def help(self):
        from ..report import html_table
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Measurement guide")
        dialog.resize(850, 580)
        view = QTextBrowser()
        view.setHtml("<h2>BiFooof measurements</h2>" + html_table([{"measure": key, **value} for key, value in MEASURES.items()]) + "<p>EC minus EO is the peak-height and band-power direction. Exponent change is EO minus EC. Missing peaks remain missing. Profile settings, usable signal, residuals, and epoch support help assess a measurement.</p>")
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        dialog.exec()

    def demo(self):
        import numpy as np
        from ..project import load_verified_source
        self.autosave()
        rate = 256
        times = np.arange(64 * rate) / rate
        generator = np.random.default_rng(53)
        sources = {}
        for condition, amplitude in (("ec", 20), ("eo", 8)):
            data = np.vstack([amplitude * np.sin(2 * np.pi * (10 + index * .1) * times) + generator.normal(0, 3, times.size) for index in range(4)])
            path = state_directory() / "demo" / f"synthetic-{condition}.json"
            atomic_json(path, {"sfreq": rate, "unit": "uV", "channel_names": ["O1", "O2", "Cz", "Fz"], "data": data.tolist()}, backup=False)
            sources[condition] = load_verified_source(path)
        state = ReviewState()
        state.change(notes="Synthetic demonstration: 10 Hz oscillation with noise, stronger in eyes closed. These are generated signals.", alias="Synthetic demonstration")
        project = make_project(sources, self.window.spectrum.settings, self.window.fit_bar.settings(), state)
        self.restore_project(project, (sources, None))
        self.changed()

    def batch(self):
        from .batch_dialog import BatchDialog
        self.batch_dialog = BatchDialog(self)
        self.batch_dialog.show()

    def report_snapshot(self):
        from ..export import _package_version
        self.flush_notes()
        window = self.window
        pictures = []
        for panel, title, ready in ((window.trace, "Traces", window.recording is not None),
            (window.spectrum, "Spectrum", window.spectrum.fit is not None), (window.compare, "Comparison", window.compare.ready),
            (window.cycles, "Cycles", window.cycles.fit is not None), (window.time, "Time", window.time.image_ready)):
            if ready:
                fig = panel.canvas.fig
                original_size = fig.get_size_inches().copy()
                try:
                    # Hidden splitter panes can be only a few pixels wide. Reports
                    # render at a fixed, readable size independent of widget geometry.
                    fig.set_size_inches(10, 6, forward=False)
                    panel.canvas.draw()
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", dpi=110, facecolor="white")
                    pictures.append({"title": title, "data": base64.b64encode(buffer.getvalue()).decode("ascii"), "width": 1100., "height": 660.})
                finally:
                    fig.set_size_inches(*original_size, forward=False)
                    panel.canvas.draw_idle()
        left, right, *_labels = window.compare._names()
        return {**deepcopy(self.state.value), "settings": asdict(window.spectrum.settings), "findings": window.findings.plain,
            "condition_labels": {"ec": left, "eo": right},
            "sources": [{"condition": condition, "path": source.source_path} for condition, source in window.sources.items()],
            "quality": quality_rows(window.recordings, window.spectrum.fits, self.state.value["reviewed"]),
            "checks": comparison_checks(window.recordings, window.spectrum.fits), "plots": pictures,
            "versions": {name: _package_version(name) for name in ("numpy", "scipy", "mne", "specparam", "bycycle")}}

    def report(self):
        if self.window._running_workers(*self.owners()):
            self.window._tell("Wait for analysis to finish, or cancel it, before exporting one consistent report.")
            return
        if not self.window.recordings:
            return
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Preview report and results")
        dialog.resize(1000, 780)
        layout = QVBoxLayout(dialog)
        options = {}
        for name, title in (("notes", "Reviewer observations"), ("quality", "Quality and comparison checks"),
            ("results", "Calculated findings and processing settings"), ("plots", "Selected current plots"), ("paths", "Include identifying source paths and filenames")):
            box = QCheckBox(title)
            box.setChecked(name != "paths")
            options[name] = box
            layout.addWidget(box)
        snapshot = self.report_snapshot()
        plot_choices = {}
        for picture in snapshot["plots"]:
            box = QCheckBox("Include " + picture["title"])
            box.setChecked(True)
            plot_choices[picture["title"]] = box
            layout.addWidget(box)
        preview = QTextBrowser()
        layout.addWidget(preview, 1)
        def refresh():
            preview.setHtml(report_html(snapshot, {**{key: box.isChecked() for key, box in options.items()}, "plot_titles": [title for title, box in plot_choices.items() if box.isChecked()]}))
        for box in options.values():
            box.toggled.connect(refresh)
        for box in plot_choices.values():
            box.toggled.connect(refresh)
        refresh()
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            self._report_options = {key: box.isChecked() for key, box in options.items()}
            self._report_options["plot_titles"] = [title for title, box in plot_choices.items() if box.isChecked()]
            try:
                self.export_results()
            finally:
                self._report_options = None

    def write_review_outputs(self, folder):
        from ..export import write_csv
        folder = Path(folder)
        written = [atomic_json(folder / "review.json", {"format": "bifooof-review", "version": 1, **self.state.value}, backup=False),
            atomic_json(folder / "measurement_dictionary.json", {"version": 1, "measures": MEASURES}, backup=False)]
        events, seen = [], set()
        for condition, source in self.window.sources.items():
            original = self.window.continuous or source
            if original.source_path in seen:
                continue
            seen.add(original.source_path)
            assignment = "continuous" if self.window.continuous is not None else condition
            events.extend({"condition": assignment, **item} for item in event_table(original, self.state.value["events"].get(original.source_path)))
        for name, rows, columns in (("events.tsv", events, ["onset", "duration", "trial_type", "channel", "condition", "state", "direction"]),
            ("channels.tsv", self.review.import_rows, ["condition", "original", "normalized", "included", "analysis_eligible", "analysis_bad", "units", "type"])):
            path = folder / name
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
        written.append(atomic_json(folder / "events.json", {"onset": {"Units": "s", "Description": "Seconds from sample zero of the original source"}, "duration": {"Units": "s"}, "channel": {"Delimiter": ","}, "condition": {"Description": "Source assignment: ec, eo, or continuous. Event eye-state classification is in state."}, "Description": "Review event sidecar, not a complete BIDS dataset"}, backup=False))
        written.append(write_csv(folder / "quality.csv", self.review.quality_rows))
        written.append(write_csv(folder / "regions.csv", regional_rows(self.window.spectrum.fits, self.state.value["regions"], self.window.spectrum.settings.alpha_band())))
        written.append(write_csv(folder / "epochs.csv", [row for condition, fit in self.window.spectrum.fits.items() for row in epoch_rows(fit, condition)]))
        if self._report_options is not None:
            written.extend(write_report(folder, self.report_snapshot(), self._report_options))
        return written

    def owners(self):
        base = (self.window, self.window.spectrum, self.window.cycles, self.window.time, self.window.compare, self, self.study)
        return (*base, self.batch_dialog) if hasattr(self, "batch_dialog") else base

    def prepare_close(self):
        self.flush_notes()
        self.autosave()
        self.preferences.update(directory=self.window._directory, geometry=bytes(self.window.saveGeometry()).hex(), import_policy=self.import_policy)
        try:
            atomic_json(self._preferences_path, self.preferences)
        except OSError:
            pass
