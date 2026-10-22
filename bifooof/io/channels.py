"""Channel-name cleaning and 10-20 standardization.

Reference suffixes used by clinical amp exports are stripped, and legacy
temporal names are stored as the modern 10-20 labels. T3, T4, T5, and T6
remain accepted aliases on the way in.
"""

from __future__ import annotations

import re

# Longer patterns are listed first. Matching is case-insensitive.
_SUFFIXES = [
    "-REF", "-LE", "-RE", "-M1", "-M2", "-A1", "-A2", "-Av", "-AV",
    "-avg", "-AVG", "-CAR", "-Ref", "-ref",
]

_PREFIXES = ["EEG ", "eeg ", "EEG", "POL ", "POL"]

OLD_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

STANDARD_19 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

_AUX_TOKENS = (
    "annotation", "annotations", "label", "status", "event", "events",
    "marker", "markers", "stim", "stimuli",
)


def clean_channel_name(name: str) -> str:
    """Return a channel name in standard modern 10-20 form."""
    n = str(name).strip()

    for pref in _PREFIXES:
        if n.upper().startswith(pref.upper()):
            n = n[len(pref):].strip()
            break

    for suf in _SUFFIXES:
        n = re.sub(re.escape(suf), "", n, flags=re.IGNORECASE)
    n = n.strip().strip("-").strip()

    up = n.upper()
    if up == "FPZ":
        return "Fpz"
    if up in ("FP1", "FP2"):
        return "Fp" + up[2:]
    if len(up) == 2 and up[1] == "Z":
        return up[0] + "z"
    if len(up) >= 2:
        result = up[0] + up[1:].lower()
        return OLD_TO_MODERN.get(result, result)
    return n


def is_auxiliary(name: str) -> bool:
    """True for annotation, marker, and status channels."""
    text = str(name).strip().lower()
    return any(token in text for token in _AUX_TOKENS)


def match_standard_19(cleaned: list[str]) -> list[str]:
    """Return which of the standard 19 channels are present, in scalp order."""
    present = set(cleaned)
    return [ch for ch in STANDARD_19 if ch in present]
