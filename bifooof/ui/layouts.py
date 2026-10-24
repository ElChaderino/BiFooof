"""Compact control groups that wrap as the desktop window gets narrower."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLayout, QSizePolicy, QWidget, QDoubleSpinBox


class CompactDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):  # noqa: N802
        text = f"{value:.{self.decimals()}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text


class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations()

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect, measure):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, height = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            if x > area.x() and x + size.width() > area.right() + 1:
                x = area.x()
                y += height + self.spacing()
                height = 0
            if not measure:
                item.setGeometry(QRect(QPoint(x, y), size))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        return y + height - rect.y() + margins.bottom()


def control_group(label: str, *widgets) -> QWidget:
    group = QWidget()
    group.setObjectName("ControlGroup")
    group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    if label:
        caption = QLabel(label)
        caption.setObjectName("H2")
        layout.addWidget(caption)
        if widgets:
            caption.setBuddy(widgets[0])
    for widget in widgets:
        layout.addWidget(widget)
    return group


class ElidedLabel(QLabel):
    """Keep long source names from forcing the entire window wider."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text):  # noqa: N802
        self._full_text = str(text)
        self.setAccessibleName(self._full_text)
        self._elide()

    def text(self):
        # Callers reading the source label still get the full description.
        return self._full_text

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full_text, Qt.ElideMiddle, self.contentsRect().width()))
