"""Eyes-closed and eyes-open labels."""

from __future__ import annotations

import re
from pathlib import Path

CONDITIONS = ("ec", "eo")
CONDITION_LABELS = {"ec": "Eyes closed", "eo": "Eyes open"}

# A trailing index (EC1, EO2) still counts. EOG and "echo" do not.
_EYES_CLOSED = re.compile(r"\beyes\s*closed\b|\bec\d*\b")
_EYES_OPEN = re.compile(r"\beyes\s*open\b|\beo\d*\b")


def guess_condition(path: str) -> str | None:
    """Return ``ec`` or ``eo`` when the filename says so."""
    stem = Path(path).stem.lower().replace("_", " ").replace("-", " ")
    if _EYES_CLOSED.search(stem):
        return "ec"
    if _EYES_OPEN.search(stem):
        return "eo"
    return None
