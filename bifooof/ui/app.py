"""Application entry point."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("QtAgg")
from PySide6.QtWidgets import QApplication

from . import icons
from .main_window import MainWindow
from .theme import build_stylesheet


def main() -> None:
    _windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BiFooof")
    app.setOrganizationName("BiFooof")
    app.setStyle("Fusion")
    app.setWindowIcon(icons.app_icon())
    app.setStyleSheet(build_stylesheet())
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


def _windows_app_id() -> None:
    """Give the process its own taskbar identity so the BiFooof mark is used."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BiFooof.SpectralStudio")
    except Exception:
        return
