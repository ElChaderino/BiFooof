"""Stability, alpha candidates, regional summaries, events, and session trends."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QInputDialog, QFileDialog, QDialog, QFormLayout, QLineEdit, QDialogButtonBox)

from ..analysis.insights import (epoch_rows, stability_summary, run_stability, alpha_candidates, regional_rows, asymmetry)
from ..analysis.settings import matched_settings, power_unit
from ..events import event_table
from ..project import read_project, atomic_json
from ..plots import clear_figure, style_axes
from .canvas import MplCanvas
from .review_panel import table_widget, fill_table
from .layouts import FlowLayout, control_group
from .workers import CallWorker, launch


class StudyPanel(QWidget):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self._rows, self._stability, self._stability_key = [], [], None
        self._generation = 0
        self._sessions = []
        self.mode = QComboBox()
        self.mode.addItems(["Epoch stability", "Alpha candidates", "Regions", "Events", "Repeated sessions"])
        self.metric = QComboBox()
        for label, value in (("Exponent", "exponent"), ("Peak center (Hz)", "peak_hz"), ("Peak height (log10)", "peak_height"),
                             ("Burst occupancy", "burst_fraction"), ("Burst episodes", "burst_episodes")):
            self.metric.addItem(label, value)
        self.kind = QComboBox()
        self.kind.addItems(["Window results", "Median / IQR / halves"])
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(8, 600)
        self.seconds.setValue(30)
        self.seconds.setSuffix(" s")
        self.compute = QPushButton("Compute nonoverlapping windows")
        self.compute.clicked.connect(self.compute_stability)
        self.search_low, self.search_high, self.half_width = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for box, value in ((self.search_low, 7), (self.search_high, 14), (self.half_width, 2)):
            box.setRange(.1, 100)
            box.setDecimals(2)
            box.setValue(value)
            box.setSuffix(" Hz")
        self.accept = QPushButton("Use selected candidate band")
        self.accept.clicked.connect(self.accept_band)
        self.region = QComboBox()
        self.left, self.right = QComboBox(), QComboBox()
        self.add_region = QPushButton("Add/edit region")
        self.add_region.clicked.connect(self.edit_region)
        self.delete_region = QPushButton("Delete region")
        self.delete_region.clicked.connect(self.remove_region)
        self.asymmetry = QComboBox()
        self.asymmetry.addItems(["Regional summaries", "Left minus right log10 PSD"])
        self.edit_event_button = QPushButton("Edit selected event")
        self.edit_event_button.clicked.connect(lambda: self.edit_event(False))
        self.add_event_button = QPushButton("Add event")
        self.add_event_button.clicked.connect(lambda: self.edit_event(True))
        self.delete_event = QPushButton("Delete event")
        self.delete_event.clicked.connect(self.remove_event)
        self.events_direction = QComboBox()
        self.events_direction.addItem("All events", "")
        self.events_direction.addItem("EC → EO", "ec_to_eo")
        self.events_direction.addItem("EO → EC", "eo_to_ec")
        self.add_sessions = QPushButton("Add saved projects")
        self.add_sessions.clicked.connect(self.load_sessions)
        self.baseline = QComboBox()
        self.baseline.setMaximumWidth(360)
        self.trend_channel = QComboBox()
        self.open_session = QPushButton("Open selected project")
        self.open_session.clicked.connect(self.open_selected_session)
        top = QHBoxLayout()
        top.addWidget(QLabel("Study"))
        top.addWidget(self.mode)
        top.addStretch()
        options = QWidget()
        flow = FlowLayout(options)
        self._groups = []
        for modes, label, widgets in (
            ((0, 4), "MEASURE", (self.metric,)), ((0,), "TABLE", (self.kind,)), ((0,), "WINDOW", (self.seconds, self.compute)),
            ((1,), "SEARCH", (self.search_low, self.search_high)), ((1,), "CHANNEL GROUP", (self.region,)),
            ((1,), "HALF WIDTH", (self.half_width, self.accept)), ((2,), "", (self.add_region, self.delete_region)),
            ((2,), "VIEW", (self.asymmetry,)), ((2,), "LEFT", (self.left,)), ((2,), "RIGHT", (self.right,)),
            ((3,), "EVENTS", (self.events_direction, self.add_event_button, self.edit_event_button, self.delete_event)),
            ((4,), "", (self.add_sessions,)), ((4,), "BASELINE", (self.baseline,)), ((4,), "", (self.open_session,))):
            group = control_group(label, *widgets)
            flow.addWidget(group)
            self._groups.append((group, modes))
        self.caption = QLabel("Select a study view. Results use the current applied settings and accepted samples.")
        self.caption.setWordWrap(True)
        self.canvas = MplCanvas(width=9, height=4)
        self.canvas.mpl_connect("pick_event", self.pick_session)
        self.canvas.setMinimumHeight(170)
        self.table = table_widget()
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(200)
        self.table.cellDoubleClicked.connect(self.inspect_row)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(options)
        layout.addWidget(self.caption)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table)
        flow.addWidget(control_group("TREND CHANNEL", self.trend_channel))
        self._groups.append((self.trend_channel.parentWidget(), (4,)))
        for combo in (self.mode, self.metric, self.kind, self.region, self.left, self.right, self.asymmetry, self.events_direction, self.baseline, self.trend_channel):
            combo.currentIndexChanged.connect(self.refresh)
        for box in (self.search_low, self.search_high):
            box.valueChanged.connect(self.refresh)

    def rows(self):
        return self._rows

    def _set_choices(self, combo, values):
        if [combo.itemText(i) for i in range(combo.count())] != values:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            if current in values:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def refresh(self, *_args):
        window, state = self.workspace.window, self.workspace.state.value
        mode = self.mode.currentIndex()
        for group, modes in self._groups:
            group.setVisible(mode in modes)
        names = list(state["regions"])
        for combo in (self.region, self.left, self.right):
            self._set_choices(combo, names)
        clear_figure(self.canvas.fig)
        ax = self.canvas.fig.add_subplot(111)
        self._rows = []
        fit = window.spectrum.fits.get(window.active)
        self.compute.setEnabled(window.recording is not None)
        bare_axes = False
        if mode == 0:
            key = (id(window.recording), window.spectrum.settings.key())
            if self._stability_key == key:
                rows = self._stability
            else:
                rows = [] if fit is None else epoch_rows(fit, window.active)
            self._rows = stability_summary(rows) if self.kind.currentIndex() else rows
            selected = [row for row in rows if row["channel"] == window._channel_name() and row.get(self.metric.currentData()) is not None]
            centers = {(row["start_s"], row["end_s"]) for row in selected}
            if len(centers) >= 2:
                values = np.array([row[self.metric.currentData()] for row in selected])
                ax.plot([(row["start_s"] + row["end_s"]) / 2 for row in selected], values, "o-", color="#1c4e78", label="Windows")
                ax.axhspan(*np.percentile(values, [25, 75]), color="#1c4e78", alpha=.1, label="IQR")
                ax.axhline(float(np.median(values)), color="#0e8f86", linestyle="--", label="Median")
                ax.legend(frameon=False, fontsize=8)
                ax.set_xlabel("Window center (s)")
                ax.set_ylabel(self.metric.currentText())
                ax.set_title(f"{window._channel_name()} · nonoverlapping window variability", loc="left", fontsize=10)
                self.caption.setText("Median/IQR and first/second halves describe variability. They are not confidence intervals. Double-click a window to inspect the signal.")
            else:
                bare_axes = True
                self.caption.setText("One fit window is not a variability series. Press Compute nonoverlapping windows. The table lists the current fit.")
        elif mode == 1:
            members = state["regions"].get(self.region.currentText(), [window._channel_name()])
            if fit and self.search_high.value() > self.search_low.value():
                self._rows = alpha_candidates(fit, members, (self.search_low.value(), self.search_high.value()))
                for item in fit.channels:
                    if item.name in members and item.freqs is not None and item.spectrum is not None:
                        ax.plot(item.freqs, item.spectrum, lw=.9, label=item.name)
                if ax.lines:
                    ax.legend(frameon=False)
                ax.axvspan(self.search_low.value(), self.search_high.value(), color="#e9f0f7")
            self.caption.setText(f"{window.active.upper()} defines the proposed band. Accepting it applies the same edges to both conditions. Select a candidate row." if self._rows else "No supported peak candidate. Inspect the spectrum or change the explicit search range/group.")
            self.accept.setEnabled(bool(self._rows))
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(_study_power(fit))
            ax.set_title("Posterior/region alpha candidates · reviewer confirmation required", loc="left", fontsize=10)
        elif mode == 2:
            rows = regional_rows(window.spectrum.fits, state["regions"], window.spectrum.settings.alpha_band())
            self._rows = asymmetry(rows, self.left.currentText(), self.right.currentText()) if self.asymmetry.currentIndex() else rows
            key = "log10_power_ratio" if self.asymmetry.currentIndex() else "mean_band_psd_uv2_hz"
            chosen = [row for row in self._rows if row.get(key) is not None]
            if chosen:
                ax.bar(np.arange(len(chosen)), [row[key] for row in chosen], color="#1c4e78")
                ax.set_xticks(np.arange(len(chosen)), [f"{row['condition']} {row.get('region', '')}" for row in chosen], rotation=20)
            ax.set_ylabel("log10(left PSD /\nright PSD)" if self.asymmetry.currentIndex() else _study_power(fit), fontsize=9)
            ax.set_xlabel("Condition and declared region")
            ax.set_title("Regional channel summaries", loc="left", fontsize=10)
            self.caption.setText("Summaries use common fitted channels across conditions. Missing members are listed. These are summaries of channel fits, not a new fit to an averaged regional spectrum.")
        elif mode == 3:
            original = window.continuous or window.sources.get(window.active)
            if original:
                rows = event_table(original, state["events"].get(original.source_path))
                direction = self.events_direction.currentData()
                self._rows = [row for row in rows if not direction or row["direction"] == direction]
                for index, row in enumerate(self._rows):
                    ax.plot([row["onset"], row["onset"] + max(.02, row["duration"])], [index, index], lw=3, color="#1c4e78")
                ax.set_yticks(range(len(self._rows)), [row["trial_type"] for row in self._rows])
            ax.set_xlabel("Original source time (s)")
            ax.set_ylabel("Event")
            ax.set_title("Reviewed events and explicit transition directions", loc="left", fontsize=10)
            self.caption.setText("Event edits are project review copies. EC→EO and EO→EC remain separate; the Time reactivity selector chooses a direction. Double-click to navigate.")
        else:
            self._refresh_sessions(ax)
        fill_table(self.table, self._rows)
        if bare_axes:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, "Press Compute nonoverlapping windows.",
                transform=ax.transAxes, ha="center", va="center", color="#63748a", fontsize=11,
            )
        else:
            style_axes(ax, grid=True)
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def compute_stability(self):
        window = self.workspace.window
        if window.recording is None:
            return
        self._generation += 1
        key = (id(window.recording), window.spectrum.settings.key())
        settings = replace(matched_settings(window.spectrum.settings, window.recording, [source for source in window.recordings.values() if source is not window.recording]))
        worker = CallWorker(self._generation, run_stability, (window.recording, settings, self.seconds.value(), window.active))
        worker.succeeded.connect(lambda gen, rows: self._stability_done(gen, key, rows))
        worker.failed.connect(lambda generation, error: self.caption.setText(error.strip().splitlines()[-1]) if generation == self._generation else None)
        launch(self, worker)
        self.caption.setText("Computing nonoverlapping spectra and cycle summaries…")

    def _stability_done(self, generation, key, rows):
        window = self.workspace.window
        if generation == self._generation and key == (id(window.recording), window.spectrum.settings.key()):
            self._stability, self._stability_key = rows, key
            self.refresh()

    def accept_band(self):
        index = self.table.currentRow()
        if not 0 <= index < len(self._rows):
            self.workspace.window._tell("Select a supported peak candidate row first.")
            return
        row = self._rows[index]
        low, high = round(row["peak_hz"] - self.half_width.value(), 3), round(row["peak_hz"] + self.half_width.value(), 3)
        window = self.workspace.window
        if low <= 0 or high >= min(source.sfreq / 2 for source in window.recordings.values()):
            self.workspace.window._tell("The proposed band must be positive and below both recordings' Nyquist frequencies.")
            return
        if any(low < fit.fmin or high > fit.fmax for fit in window.spectrum.fits.values()):
            self.workspace.window._tell("Refit both spectra to cover the proposed band before accepting it.")
            return
        origin = f"{window.active.upper()} {row['channel']} candidate {row['peak_hz']:.3f} Hz; search {self.search_low.value():g}–{self.search_high.value():g} Hz"
        self.workspace.state.change(band_origin=origin)
        window.fit_bar.set_band(low, high, named=False)
        window._on_fit_band(low, high)
        self.workspace.changed()

    def edit_region(self):
        name, accepted = QInputDialog.getText(self, "Channel region", "Region name", text=self.left.currentText())
        if not accepted or not name.strip():
            return
        members, accepted = QInputDialog.getText(self, "Region members", "Comma-separated channel names", text=", ".join(self.workspace.state.value["regions"].get(name, [])))
        if accepted:
            regions = deepcopy(self.workspace.state.value["regions"])
            regions[name.strip()] = list(dict.fromkeys(value.strip() for value in members.split(",") if value.strip()))
            self.workspace.state.change(regions=regions)
            self.workspace.changed()
            self.refresh()

    def remove_region(self):
        regions = deepcopy(self.workspace.state.value["regions"])
        regions.pop(self.left.currentText(), None)
        self.workspace.state.change(regions=regions)
        self.workspace.changed()
        self.refresh()

    def _event_source(self):
        window = self.workspace.window
        return window.continuous or window.sources.get(window.active)

    def edit_event(self, adding=False):
        source = self._event_source()
        if source is None:
            return
        index = self.table.currentRow()
        if not adding and not 0 <= index < len(self._rows):
            return
        chosen = {"onset": source.source_start_s, "duration": 0, "trial_type": "Event", "state": "", "channel": "*"} if adding else self._rows[index]
        dialog = QDialog(self)
        dialog.setWindowTitle("Reviewed event")
        form = QFormLayout(dialog)
        onset, duration = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, field in ((onset, "onset"), (duration, "duration")):
            widget.setRange(0, source.source_start_s + source.duration_s)
            widget.setDecimals(4)
            widget.setValue(chosen[field])
        label = QLineEdit(chosen["trial_type"])
        state = QComboBox()
        for text, value in (("Informational", ""), ("Eyes closed", "ec"), ("Eyes open", "eo")):
            state.addItem(text, value)
        state.setCurrentIndex(max(0, state.findData(chosen.get("state", ""))))
        channel = QLineEdit(chosen.get("channel", "*"))
        for title, widget in (("Source onset (s)", onset), ("Duration (s)", duration), ("Label", label), ("State mapping", state), ("Channel (* for all)", channel)):
            form.addRow(title, widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            if onset.value() + duration.value() > source.source_start_s + source.duration_s:
                self.workspace.window._tell("That event ends beyond the source recording.")
                return
            rows = event_table(source, self.workspace.state.value["events"].get(source.source_path))
            if not adding:
                rows.remove(chosen)
            rows.append({"onset": onset.value(), "duration": duration.value(), "trial_type": label.text(), "state": state.currentData(), "channel": channel.text()})
            self._save_events(source, rows)

    def remove_event(self):
        source = self._event_source()
        index = self.table.currentRow()
        if source and 0 <= index < len(self._rows):
            rows = event_table(source, self.workspace.state.value["events"].get(source.source_path))
            rows.remove(self._rows[index])
            self._save_events(source, rows)

    def _save_events(self, source, rows):
        events = deepcopy(self.workspace.state.value["events"])
        events[source.source_path] = event_table(source, rows)
        self.workspace.state.change(events=events)
        self.workspace.window.time.review_events = events
        self.workspace.window.time._kick()
        self.workspace.changed()
        self.refresh()

    def inspect_row(self, row, _column):
        if not 0 <= row < len(self._rows):
            return
        item = self._rows[row]
        window = self.workspace.window
        if self.mode.currentIndex() == 0 and "start_s" in item:
            self.workspace.inspect(item["condition"], item["channel"], (item["start_s"] + item["end_s"]) / 2)
        elif self.mode.currentIndex() == 3:
            source = self._event_source()
            moment = item["onset"] - window.recording.source_start_s
            if 0 <= moment < window.recording.duration_s:
                self.workspace.inspect(window.active, window._channel_name(), moment)
            else:
                self.caption.setText("This source event lies outside the currently selected condition. Switch to the condition containing this source time.")

    def load_sessions(self):
        paths, _selected = QFileDialog.getOpenFileNames(self, "Add saved projects to session index", self.workspace.window._directory, "BiFooof project (*.json)")
        if paths:
            self.workspace.preferences["session_projects"] = list(dict.fromkeys([*self.workspace.preferences.get("session_projects", []), *paths]))
            atomic_json(self.workspace._preferences_path, self.workspace.preferences)
            self.refresh()

    def _refresh_sessions(self, ax):
        window = self.workspace.window
        entries, errors = [], []
        for path in self.workspace.preferences.get("session_projects", []):
            try:
                project = read_project(path)
                review = project.get("review", {})
                for row in project.get("summaries", []):
                    entries.append({"project": path, "alias": review.get("alias", ""), "review_notes": review.get("notes", ""), "date": review.get("date") or project["saved_utc"][:10], **row})
            except (ValueError, OSError, KeyError) as exc:
                errors.append(f"{Path(path).name}: {exc}")
        current_channel = self.trend_channel.currentText()
        self._set_choices(self.trend_channel, sorted({row["channel"] for row in entries}))
        if not current_channel and window._channel_name() in [self.trend_channel.itemText(i) for i in range(self.trend_channel.count())]:
            self.trend_channel.blockSignals(True)
            self.trend_channel.setCurrentText(window._channel_name())
            self.trend_channel.blockSignals(False)
        entries = [row for row in entries if row["channel"] == self.trend_channel.currentText()]
        self._sessions = entries
        projects = list(dict.fromkeys(row["project"] for row in entries))
        self._set_choices(self.baseline, projects)
        metric = self.metric.currentData()
        rows = []
        for row in sorted(entries, key=lambda item: (item["date"], item["project"], item["condition"], item["channel"])):
            baseline = next((item for item in entries if item["project"] == self.baseline.currentText() and item["condition"] == row["condition"] and item["channel"] == row["channel"]), None)
            value = row.get(metric)
            base_value = None if baseline is None else baseline.get(metric)
            compatible = baseline is not None and row["settings"] == baseline["settings"] and row.get("reference") == baseline.get("reference") and row.get("reference_contributors") == baseline.get("reference_contributors")
            stored_unknown = row.get("reference") == "stored" and (row.get("stored_reference", "Unknown") == "Unknown" or baseline is None or row.get("stored_reference") != baseline.get("stored_reference"))
            rows.append({"project": row["project"], "date": row["date"], "alias": row["alias"], "condition": row["condition"], "channel": row["channel"],
                "measure": metric, "value": value, "baseline_delta": None if value is None or base_value is None else value - base_value,
                "comparison": "Stored reference unknown/different" if stored_unknown else "Matched settings/reference" if compatible else "Review settings/reference or missing baseline",
                "accepted_s": row.get("accepted_s"), "r_squared": row.get("r_squared"),
                "baseline_project": self.baseline.currentText(), "delta_definition": "session minus baseline",
                "profile": row.get("profile", "Custom"), "review_notes": row["review_notes"],
                "reference": row.get("reference"), "stored_reference": row.get("stored_reference", "Unknown"),
                "reference_contributors": json.dumps(row.get("reference_contributors", [])), "settings": json.dumps(row["settings"], sort_keys=True),
                })
        self._rows = rows
        sessions = list(dict.fromkeys(row["project"] for row in rows))
        for condition in ("ec", "eo"):
            points = [(sessions.index(row["project"]), row["value"]) for row in rows if row["condition"] == condition and row["value"] is not None]
            if points:
                line, = ax.plot([x for x, y in points], [y for x, y in points], "o-", label=condition.upper(), picker=6)
                line._session_projects = [sessions[x] for x, y in points]
        if ax.lines:
            ax.legend(frameon=False)
        labels = [next(row["date"] for row in rows if row["project"] == path) + f" · {index + 1}" for index, path in enumerate(sessions)]
        stride = max(1, (len(sessions) + 7) // 8)
        ax.set_xticks(list(range(0, len(sessions), stride)), labels[::stride])
        ax.set_xlabel(f"Saved sessions · {self.trend_channel.currentText()}")
        ax.set_ylabel(self.metric.currentText())
        ax.set_title("Repeated-session measurements · each point retains its project", loc="left", fontsize=10)
        self.caption.setText("Baseline Δ is session minus the chosen baseline. Double-click a plotted point or table row to open its saved project. Changes in processing/reference are flagged; this view describes measurement differences. " + " ".join(errors))

    def pick_session(self, event):
        if self.mode.currentIndex() == 4 and event.mouseevent.dblclick and len(event.ind):
            paths = getattr(event.artist, "_session_projects", [])
            if int(event.ind[0]) < len(paths):
                self.workspace.open_project(paths[int(event.ind[0])])

    def open_selected_session(self):
        index = self.table.currentRow()
        if self.mode.currentIndex() == 4 and 0 <= index < len(self._rows):
            self.workspace.open_project(self._rows[index]["project"])


def _study_power(fit) -> str:
    applied = "stored" if fit is None else str((fit.provenance or {}).get("reference_applied", "stored"))
    return f"log10 power ({power_unit(applied)})"
