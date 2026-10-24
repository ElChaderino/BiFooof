"""Review overview and journal; decisions live in ReviewState, not widgets."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QPlainTextEdit, QLineEdit, QPushButton)


def fill_table(table, rows, columns=None):
    selected = table.currentRow()
    selected_key = tuple(table.item(selected, column).data(Qt.UserRole) if selected >= 0 and table.item(selected, column) else None for column in range(min(2, table.columnCount())))
    table.blockSignals(True)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels([column.replace("_", " ").capitalize() for column in columns])
    table.setRowCount(len(rows))
    for index, row in enumerate(rows):
        for column, key in enumerate(columns):
            value = row.get(key)
            text = "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
            item = QTableWidgetItem(text)
            item.setData(Qt.UserRole, value)
            item.setToolTip(text)
            table.setItem(index, column, item)
    table.blockSignals(False)
    if selected_key and any(value is not None for value in selected_key):
        for index in range(table.rowCount()):
            if tuple(table.item(index, column).data(Qt.UserRole) for column in range(len(selected_key))) == selected_key:
                table.selectRow(index)
                break


def table_widget():
    table = QTableWidget()
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class ReviewPanel(QWidget):
    inspect_requested = Signal(str, str, float)
    action_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.quality_rows, self.bookmark_rows, self.import_rows = [], [], []
        self.heading = QLabel("Open a recording to review data support, comparison conditions, and notes.")
        self.heading.setWordWrap(True)
        self.profile_label = QLabel("Processing profile: built-in defaults")
        controls = QHBoxLayout()
        for label, action in (("Save profile", "save_profile"), ("Apply profile", "choose_profile"),
                              ("Add bookmark", "bookmark"), ("Undo review", "undo"), ("Redo review", "redo")):
            button = QPushButton(label)
            button.setObjectName("Secondary")
            button.clicked.connect(lambda _checked=False, chosen=action: self.action_requested.emit(chosen))
            controls.addWidget(button)
        controls.addStretch()
        self.pages = QTabWidget()
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        self.checks = QPlainTextEdit()
        self.checks.setReadOnly(True)
        self.checks.setMaximumHeight(80)
        self.quality = table_widget()
        self.quality.cellDoubleClicked.connect(lambda row, _column: self.inspect_quality(row))
        actions = QHBoxLayout()
        for label, action in (("Inspect channel", "inspect"), ("Mark reviewed", "reviewed"),
                              ("Exclude channel", "exclude_channel"), ("Restore channel", "restore_channel")):
            button = QPushButton(label)
            button.setObjectName("Secondary")
            button.clicked.connect(lambda _checked=False, chosen=action: self.action_requested.emit(chosen))
            actions.addWidget(button)
        actions.addStretch()
        overview_layout.addWidget(self.checks)
        overview_layout.addWidget(self.quality, 1)
        overview_layout.addLayout(actions)
        self.pages.addTab(overview, "Quality and comparison")
        journal = QWidget()
        journal_layout = QVBoxLayout(journal)
        fields = QHBoxLayout()
        self.alias, self.date = QLineEdit(), QLineEdit()
        self.alias.setPlaceholderText("Participant alias (optional)")
        self.date.setPlaceholderText("Session date, YYYY-MM-DD")
        fields.addWidget(self.alias)
        fields.addWidget(self.date)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Reviewer observations. These remain separate from calculated findings.")
        self.notes.setMaximumHeight(160)
        self.bookmarks = table_widget()
        self.bookmarks.cellDoubleClicked.connect(self.inspect_bookmark)
        row = QHBoxLayout()
        for label, action in (("Previous", "previous_bookmark"), ("Next", "next_bookmark"),
                              ("Edit selected", "edit_bookmark"), ("Delete selected", "delete_bookmark"),
                              ("Exclude selected interval", "exclude_bookmark")):
            button = QPushButton(label)
            button.setObjectName("Secondary")
            button.clicked.connect(lambda _checked=False, chosen=action: self.action_requested.emit(chosen))
            row.addWidget(button)
        journal_layout.addLayout(fields)
        journal_layout.addWidget(self.notes)
        journal_layout.addWidget(self.bookmarks, 1)
        journal_layout.addLayout(row)
        self.pages.addTab(journal, "Notes and bookmarks")
        imports = QWidget()
        imports_layout = QVBoxLayout(imports)
        self.import_summary = QLabel()
        self.import_summary.setWordWrap(True)
        self.imports = table_widget()
        imports_layout.addWidget(self.import_summary)
        imports_layout.addWidget(self.imports, 1)
        self.pages.addTab(imports, "Import inspection")
        root = QVBoxLayout(self)
        root.addWidget(self.heading)
        root.addWidget(self.profile_label)
        root.addLayout(controls)
        root.addWidget(self.pages, 1)

    def inspect_quality(self, row):
        if 0 <= row < len(self.quality_rows):
            item = self.quality_rows[row]
            self.inspect_requested.emit(item["condition"], item["channel"], -1)

    def inspect_bookmark(self, row, _column=0):
        if 0 <= row < len(self.bookmark_rows):
            item = self.bookmark_rows[row]
            self.inspect_requested.emit(item["condition"], item.get("channel", ""), item["time_s"])

    def rows(self):
        return self.quality_rows if self.pages.currentIndex() == 0 else self.bookmark_rows if self.pages.currentIndex() == 1 else self.import_rows

    def refresh(self, recordings, fits, state):
        from ..analysis.insights import quality_rows, comparison_checks
        self.quality_rows = quality_rows(recordings, fits, state.value["reviewed"])
        fill_table(self.quality, self.quality_rows, ["condition", "channel", "review", "accepted_s", "excluded_percent", "flags", "r_squared", "fit_error", "welch_segments", "epochs", "reference_requested", "reference_applied"])
        self.checks.setPlainText("\n".join(f"{row['level'].upper()} · {row['check']}: {row['detail']}" for row in comparison_checks(recordings, fits)))
        self.bookmark_rows = state.value["bookmarks"]
        fill_table(self.bookmarks, self.bookmark_rows, ["condition", "time_s", "duration_s", "channel", "label", "note"])
        self.import_rows = []
        summaries = []
        for condition, source in recordings.items():
            info = source.import_info
            originals = info.get("original_channels", source.ch_names)
            normalized = info.get("normalized_channels", source.ch_names)
            for original, cleaned in zip(originals, normalized):
                self.import_rows.append({"condition": condition, "original": original, "normalized": cleaned,
                    "included": cleaned in source.ch_names, "analysis_bad": cleaned in source.bad_channels,
                    "analysis_eligible": cleaned in source.ch_names and source.analysis_channel(cleaned),
                    "units": info.get("source_units", {}).get(original, "Unknown") if isinstance(info.get("source_units"), dict) else info.get("source_units", "Unknown"),
                    "type": info.get("channel_types", {}).get(original, "Unknown")})
            summaries.append(f"{condition.upper()}: {source.sfreq:g} Hz; {source.duration_s:g} s; channel policy {info.get('channel_policy', 'standard')}; stored reference {info.get('stored_reference', 'Unknown')}.")
        fill_table(self.imports, self.import_rows)
        self.import_summary.setText("\n".join(summaries))
        self.heading.setText(f"{len(recordings)} recording(s) · {len(self.quality_rows)} channel results · {len(state.value['reviewed'])} review decisions. Double-click a row to inspect its evidence.")

    def restore_notes(self, state):
        for widget, key in ((self.notes, "notes"), (self.alias, "alias"), (self.date, "date")):
            widget.blockSignals(True)
            widget.setPlainText(state.value[key]) if widget is self.notes else widget.setText(state.value[key])
            widget.blockSignals(False)
