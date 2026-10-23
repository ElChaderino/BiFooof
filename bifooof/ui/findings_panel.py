"""Clinician-facing reading of the fits already on the other pages."""

from __future__ import annotations

import html

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from ..analysis.cycles import CycleFit
from ..analysis.findings import ReportSection, compile_report, report_text
from ..analysis.settings import FitSettings
from ..analysis.specparam_fit import SpectrumFit
from ..io.recording import Recording
from .theme import ACCENT, INK, LINE, MUTED, PANEL


class FindingsPanel(QWidget):
    """Scrollable report. Copy puts the same words on the clipboard."""

    def __init__(self):
        super().__init__()
        self.sections: list[ReportSection] = []
        self.plain = ""

        title = QLabel("Findings")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {ACCENT};")
        hint = QLabel(
            "A reading of the current fit. The channel follows whatever you select on the other pages."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("Secondary")
        self.copy_button.setToolTip("Copy this page as plain text for a note.")
        self.copy_button.clicked.connect(self.copy)

        header = QHBoxLayout()
        header.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(title)
        text.addWidget(hint)
        header.addLayout(text, 1)
        header.addWidget(self.copy_button)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setStyleSheet(
            f"QTextBrowser {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE}; "
            f"border-radius: 10px; padding: 14px 18px; font-size: 14px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.browser, 1)
        self.set_report([ReportSection("Findings", (
            "Open a recording. This page fills in when the spectrum fit finishes.",
        ))])

    def update_report(
        self,
        recordings: dict[str, Recording],
        spectra: dict[str, SpectrumFit],
        cycles: dict[str, CycleFit],
        band: tuple[float, float],
        channel: str,
        settings: FitSettings | None = None,
        pair: tuple[str, str] | None = None,
    ) -> None:
        self.set_report(compile_report(recordings, spectra, cycles, band, channel, settings, pair))

    def set_report(self, sections: list[ReportSection]) -> None:
        self.sections = list(sections)
        self.plain = report_text(sections)
        self.browser.setHtml(_html(sections))

    def rows(self) -> list[dict]:
        return [
            {"section": section.title, "text": line}
            for section in self.sections
            for line in section.lines
        ]

    def copy(self) -> None:
        QGuiApplication.clipboard().setText(self.plain)
        self.copy_button.setText("Copied")
        QTimer.singleShot(1200, lambda: self.copy_button.setText("Copy"))


def _html(sections: list[ReportSection]) -> str:
    parts = [
        "<html><body style='font-family: Segoe UI, Arial, sans-serif; "
        f"color: {INK}; line-height: 1.45;'>"
    ]
    for section in sections:
        parts.append(
            f"<h2 style='font-size: 15px; color: {ACCENT}; margin: 16px 0 6px 0;'>"
            f"{html.escape(section.title)}</h2>"
        )
        for line in section.lines:
            color = MUTED if section.title == "What this page does not say" else INK
            parts.append(
                f"<p style='margin: 0 0 8px 0; color: {color};'>{html.escape(line)}</p>"
            )
    parts.append("</body></html>")
    return "".join(parts)
