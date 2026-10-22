"""Vector icons in the same stroke style as the other EEG tools.

Glyphs are 24×24, round-capped strokes, drawn through Qt's SVG renderer so the
window icon and the toolbar stay sharp on a high-DPI screen. The BiFooof mark
is the shared disc, with one tall peak instead of the even NeuroDx trace.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{c}" stroke-width="{w}" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)

_BODIES: dict[str, str] = {
    # Disc plus a trace that rises into one spectral peak.
    "logo": (
        '<circle cx="12" cy="12" r="9.2"/>'
        '<path d="M5.2 13.4h1.5l1.15-2.1 1.25 2.1 1.7-6.6 1.7 6.6 1.25-2.1 1.15 2.1H18.8"/>'
    ),
    "folder": (
        '<path d="M3.2 7.2a1.8 1.8 0 0 1 1.8-1.8h3.6l2 2.3h8.4a1.8 1.8 0 0 1 '
        '1.8 1.8v8.5a1.8 1.8 0 0 1-1.8 1.8H5a1.8 1.8 0 0 1-1.8-1.8z"/>'
    ),
    "image": (
        '<rect x="4" y="4.6" width="16" height="14.8" rx="2"/>'
        '<circle cx="9" cy="9.4" r="1.3"/>'
        '<path d="M4.6 15.6 8.6 12.2l2.8 2.2 3-3.2 4.8 4.2"/>'
    ),
    "table": (
        '<rect x="3.8" y="5" width="16.4" height="14" rx="1.8"/>'
        '<path d="M3.8 10h16.4"/><path d="M3.8 14.5h16.4"/><path d="M10 5v14"/>'
    ),
    "layers": (
        '<path d="M12 3.4 3 8l9 4.5L21 8z"/><path d="M3 12l9 4.5L21 12"/>'
        '<path d="M3 16l9 4.5L21 16"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="9"/><path d="M12 11.2v4.8"/>'
        '<path d="M12 8h.02"/>'
    ),
    "waveform": '<path d="M3 12h2.4l1.4-5 2.2 11 2-9 1.6 6 1.4-3h4"/>',
    "activity": '<path d="M3 12h3.2l2.4-6.8 4 13.6 2.4-6.8H21"/>',
    "cycles": '<path d="M3 14c2.1-5.2 4.1-5.2 6.2 0s4.1 5.2 6.2 0 4.1-5.2 6.2 0"/>',
    "compare": (
        '<rect x="3.5" y="5" width="7.2" height="14" rx="1.6"/>'
        '<rect x="13.3" y="5" width="7.2" height="14" rx="1.6"/>'
    ),
    "chevron-down": '<path d="M6 9.5 12 15l6-5.5"/>',
}

TAB_ICONS = {
    "Traces": "waveform",
    "Spectrum": "activity",
    "Cycles": "cycles",
    "Compare": "compare",
}


def _svg_bytes(name: str, color: str, width: float = 1.8) -> QByteArray:
    svg = _TEMPLATE.format(c=color, w=width, body=_BODIES[name])
    return QByteArray(svg.encode("utf-8"))


@lru_cache(maxsize=256)
def pixmap(name: str, color: str = "#1c4e78", size: int = 20,
           width: float = 1.8, dpr: float = 2.0) -> QPixmap:
    """Render a glyph to a hi-DPI pixmap."""
    renderer = QSvgRenderer(_svg_bytes(name, color, width))
    px = int(round(size * dpr))
    image = QPixmap(px, px)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()
    image.setDevicePixelRatio(dpr)
    return image


@lru_cache(maxsize=256)
def icon(name: str, color: str = "#1c4e78", size: int = 20, width: float = 1.8) -> QIcon:
    """Return a recolored toolbar icon."""
    return QIcon(pixmap(name, color, size, width))


def app_icon() -> QIcon:
    """Window and taskbar icon at the sizes Windows asks for."""
    mark = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        stroke = 2.15 if size <= 24 else 1.7
        mark.addPixmap(pixmap("logo", "#1c4e78", size, width=stroke, dpr=1.0))
    return mark


def save_app_icon(folder: str | Path) -> Path:
    """Write a multi-size ``.ico`` and a 256 px PNG of the BiFooof mark."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    sizes = (16, 24, 32, 48, 64, 128, 256)
    blobs = {size: _png_bytes(size) for size in sizes}
    (destination / "bifooof-256.png").write_bytes(blobs[256])
    ico = destination / "bifooof.ico"
    ico.write_bytes(_ico_bytes(blobs))
    return ico


def prepare_qss_assets(cache_dir: Path) -> dict[str, str]:
    """Write the chevron the stylesheet can only load from a file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _export_png("chevron-down", "#5b6b7a", 32, cache_dir / "chevron.png")
    return {"chevron": str(path).replace("\\", "/")}


def _png_bytes(size: int) -> bytes:
    stroke = 2.15 if size <= 24 else 1.7
    renderer = QSvgRenderer(_svg_bytes("logo", "#1c4e78", stroke))
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def _export_png(name: str, color: str, size: int, out_path: Path, width: float = 2.0) -> Path:
    renderer = QSvgRenderer(_svg_bytes(name, color, width))
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path), "PNG")
    return out_path


def _ico_bytes(blobs: dict[int, bytes]) -> bytes:
    """Vista-style icon: PNG frames, so the 256 px size stays sharp."""
    sizes = list(blobs)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    entries = bytearray()
    images = bytearray()
    offset = 6 + 16 * len(sizes)
    for size in sizes:
        data = blobs[size]
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        images += data
        offset += len(data)
    return header + bytes(entries) + bytes(images)
