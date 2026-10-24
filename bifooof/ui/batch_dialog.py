"""An explicit file-to-condition queue using a locked processing snapshot."""
from dataclasses import replace
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QFileDialog, QTableWidgetItem, QMessageBox)
from ..batch_queue import run_queue
from .review_panel import table_widget
from .workers import CallWorker, launch


class BatchDialog(QDialog):
    item_progress = Signal(int, str, str)

    def __init__(self, workspace):
        super().__init__(workspace.window)
        self.workspace = workspace
        self.settings = replace(workspace.window.spectrum.settings)
        self.setWindowTitle("Batch queue · applied processing settings locked")
        self.resize(1000, 620)
        self._closing = False
        self._live_workers = []
        self.item_progress.connect(self.update_item)
        self.caption = QLabel(f"Locked snapshot: {self.settings.fmin:g}–{self.settings.fmax:g} Hz, {self.settings.reference} reference, {self.settings.alpha_low:g}–{self.settings.alpha_high:g} Hz band. Assign each recording's condition.")
        self.caption.setWordWrap(True)
        self.table = table_widget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Recording", "Condition", "Status", "Output"])
        self.group = QComboBox()
        self.group.addItem("Preferred cycle channel", None)
        for name, members in workspace.state.value["regions"].items():
            self.group.addItem(name, list(members))
        self.output = None
        row = QHBoxLayout()
        self.add_button = QPushButton("Add recordings")
        self.add_button.clicked.connect(self.add_files)
        self.run_button = QPushButton("Run pending / retry failed")
        self.run_button.clicked.connect(self.start)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel)
        for widget in (self.add_button, self.group, self.run_button, cancel):
            row.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addWidget(self.caption)
        layout.addLayout(row)
        layout.addWidget(self.table, 1)

    def add_files(self):
        from ..io.sniff import FILE_FILTER
        paths, _selected = QFileDialog.getOpenFileNames(self, "Batch recordings", self.workspace.window._directory, FILE_FILTER)
        for path in paths:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(path))
            choice = QComboBox()
            choice.addItems(["ec", "eo"])
            self.table.setCellWidget(row, 1, choice)
            self.table.setItem(row, 2, QTableWidgetItem("Pending"))
            self.table.setItem(row, 3, QTableWidgetItem(""))

    def start(self):
        items = [{"index": row, "path": self.table.item(row, 0).text(), "condition": self.table.cellWidget(row, 1).currentText()}
            for row in range(self.table.rowCount()) if self.table.item(row, 2).text() != "Complete"]
        if not items:
            return
        if not self.output:
            self.output = QFileDialog.getExistingDirectory(self, "Batch output folder", self.workspace.window._directory)
        if not self.output:
            return
        self.run_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.group.setEnabled(False)
        worker = CallWorker(1, run_queue, (items, replace(self.settings), self.output, self.group.currentData(), self.item_progress.emit))
        worker.succeeded.connect(lambda _generation, rows: self.caption.setText(f"Batch finished: {sum(row['status'] == 'Complete' for row in rows)} completed. Aggregate tables are in {self.output}."))
        worker.failed.connect(lambda _generation, message: self.caption.setText(message.strip().splitlines()[-1]))
        worker.finished.connect(self.finished_work)
        launch(self, worker)

    def update_item(self, row, status, output):
        self.table.item(row, 2).setText(status)
        self.table.item(row, 3).setText(output)

    def cancel(self):
        for worker in self._live_workers:
            worker.cancel()
        self.caption.setText("Cancellation requested; the current library call stops at its next checkpoint. Completed items keep their outputs.")

    def finished_work(self):
        self.run_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.group.setEnabled(True)
        if self._closing:
            self.close()

    def closeEvent(self, event):
        if any(worker.isRunning() for worker in self._live_workers):
            self._closing = True
            self.cancel()
            event.ignore()
            return
        event.accept()
