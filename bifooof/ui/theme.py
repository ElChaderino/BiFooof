"""Clinical desktop theme shared with the other EEG tools."""

from __future__ import annotations

from pathlib import Path

INK = "#16232f"
MUTED = "#63748a"
FAINT = "#8b99a8"
ACCENT = "#1c4e78"
ACCENT_HOVER = "#245f92"
ACCENT_PRESS = "#163f61"
ACCENT_SOFT = "#e9f0f7"
TEAL = "#0e8f86"
LINE = "#e4eaf1"
LINE_STRONG = "#d2dbe6"
BG = "#edf1f6"
PANEL = "#ffffff"
CARD = "#f7fafc"
WARN = "#b3261e"


def _stylesheet(chevron: str) -> str:
    arrow = (
        f'QComboBox::down-arrow {{ image: url("{chevron}"); width: 14px; height: 14px; }}'
        if chevron else ""
    )
    return f"""
    * {{ outline: 0; }}
    QMainWindow, QWidget {{
        background: {BG}; color: {INK};
        font-family: "Segoe UI", Arial, sans-serif; font-size: 13px;
    }}
    QToolTip {{
        background: {INK}; color: #ffffff; border: none; padding: 6px 9px;
        border-radius: 6px; font-size: 12px;
    }}
    QFrame#Header {{ background: {PANEL}; border: none; border-bottom: 1px solid {LINE}; }}
    QLabel#Brand {{ font-size: 20px; font-weight: 700; color: {ACCENT}; letter-spacing: 0.3px; }}
    QLabel#BrandSub {{ font-size: 11px; color: {MUTED}; letter-spacing: 1.4px; }}
    QLabel#H2 {{ font-size: 11px; font-weight: 700; color: {MUTED}; letter-spacing: 1.1px; }}
    QLabel#Muted {{ color: {MUTED}; }}
    QLabel#Faint {{ color: {FAINT}; font-size: 11px; }}
    QPushButton {{
        background: {ACCENT}; color: white; border: none; border-radius: 8px;
        padding: 8px 14px; font-weight: 600;
    }}
    QPushButton:hover {{ background: {ACCENT_HOVER}; }}
    QPushButton:pressed {{ background: {ACCENT_PRESS}; }}
    QPushButton:disabled {{ background: #c3ccd7; color: #eef2f6; }}
    QPushButton#Secondary {{
        background: {PANEL}; color: {ACCENT}; border: 1px solid {LINE_STRONG};
    }}
    QPushButton#Secondary:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
    QPushButton#Secondary:disabled {{ background: {PANEL}; color: #b3c0cd; border-color: {LINE}; }}
    QPushButton#Ghost {{
        background: transparent; color: {INK}; border: none; border-radius: 8px;
        padding: 6px; font-weight: 600;
    }}
    QPushButton#Ghost:hover {{ background: {ACCENT_SOFT}; }}
    QComboBox, QLineEdit {{
        background: {PANEL}; border: 1px solid {LINE_STRONG}; border-radius: 8px;
        padding: 6px 10px; min-height: 18px;
        selection-background-color: {ACCENT_SOFT}; selection-color: {INK};
    }}
    QComboBox:hover, QComboBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{
        subcontrol-origin: padding; subcontrol-position: center right;
        width: 26px; border: none;
    }}
    {arrow}
    QComboBox QAbstractItemView {{
        background: {PANEL}; border: 1px solid {LINE_STRONG}; border-radius: 8px;
        padding: 4px; selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT};
        outline: 0;
    }}
    QListWidget {{
        background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 4px;
    }}
    QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
    QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
    QListWidget::item:hover {{ background: {BG}; }}
    QTabWidget::pane {{
        border: 1px solid {LINE}; background: {PANEL}; border-radius: 12px; top: -1px;
        padding: 8px;
    }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent; color: {MUTED}; padding: 9px 16px 11px 16px;
        margin-right: 2px; border: none; border-bottom: 2px solid transparent; font-weight: 600;
    }}
    QTabBar::tab:hover {{ color: {ACCENT}; }}
    QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
    QTableView, QTableWidget {{
        background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px;
        gridline-color: {LINE}; alternate-background-color: {CARD};
        selection-background-color: {ACCENT_SOFT}; selection-color: {INK};
    }}
    QTableView::item, QTableWidget::item {{ padding: 3px 6px; }}
    QHeaderView::section {{
        background: {CARD}; color: {MUTED}; border: none;
        border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE_STRONG};
        padding: 7px 8px; font-weight: 700;
    }}
    QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {LINE}; }}
    QStatusBar::item {{ border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: #c3ccd7; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: #c3ccd7; border-radius: 5px; min-width: 30px; }}
    QScrollBar::handle:horizontal:hover {{ background: {MUTED}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QDialog {{ background: {BG}; }}
    QMessageBox {{ background: {PANEL}; }}
    """


def build_stylesheet(cache_dir: Path | None = None) -> str:
    """Return the application stylesheet, including the combobox chevron."""
    from . import icons

    if cache_dir is None:
        cache_dir = Path.home() / ".bifooof" / "ui_cache"
    try:
        chevron = icons.prepare_qss_assets(cache_dir)["chevron"]
    except Exception:
        chevron = ""
    return _stylesheet(chevron)
