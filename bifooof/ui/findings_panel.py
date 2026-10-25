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
        hint = QLabel("The channel is the one selected on the other pages. These are the same numbers.")
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
            "border-radius: 12px; padding: 6px 4px; }"
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
        rows = []
        for section in self.sections:
            rows.extend(
                {"section": section.title, "text": line}
                for line in section.lines
                if line.strip()
            )
            if section.aside:
                rows.append({"section": section.title, "text": section.aside})
        return rows

    def copy(self) -> None:
        QGuiApplication.clipboard().setText(self.plain)
        self.copy_button.setText("Copied")
        QTimer.singleShot(1200, lambda: self.copy_button.setText("Copy"))


def _html(sections: list[ReportSection]) -> str:
    quiet = "What this page does not say"
    parts = [
        "<html><body style=\"font-family: Cambria, 'Palatino Linotype', Georgia, serif; "
        f"color: {INK}; font-size: 16px; line-height: 1.55; margin: 4px 18px 16px 18px;\">"
    ]
    for index, section in enumerate(sections):
        top = "4px" if index == 0 else "22px"
        parts.append(
            "<h2 style=\"font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; "
            f"font-weight: 700; letter-spacing: 0.08em; color: {ACCENT}; "
            f"margin: {top} 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid {LINE};\">"
            f"{html.escape(section.title)}</h2>"
        )
        muted = section.title == quiet
        for line in section.lines:
            parts.append(_paragraph(line, muted))
        if section.aside:
            parts.append(
                "<p style=\"font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; "
                f"font-style: italic; color: {MUTED}; margin: 14px 0 2px 0;\">"
                f"{html.escape(section.aside)}</p>"
            )
    parts.append("</body></html>")
    return "".join(parts)


def _paragraph(line: str, muted: bool) -> str:
    if not line.strip():
        return "<div style='height: 8px;'></div>"
    if _is_label(line):
        return (
            "<p style=\"font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; "
            f"font-weight: 600; color: {ACCENT}; margin: 12px 0 4px 0;\">{html.escape(line)}</p>"
        )
    color = MUTED if muted else INK
    split = _split_label(line)
    if split is None:
        return f"<p style='margin: 0 0 8px 0; color: {color};'>{html.escape(line)}</p>"
    head, tail = split
    return (
        f"<p style='margin: 0 0 8px 0; color: {color};'>"
        "<span style=\"font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600;\">"
        f"{html.escape(head)}</span> {html.escape(tail)}</p>"
    )


def _is_label(line: str) -> bool:
    return ":" not in line and len(line) <= 56 and line[-1:] not in ".;"


def _split_label(line: str) -> tuple[str, str] | None:
    head, sep, tail = line.partition(": ")
    if not sep or not tail or len(head) > 42 or head.count(" ") > 6:
        return None
    return head, tail
