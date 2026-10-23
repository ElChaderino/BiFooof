"""WinEEG marks that name an eyes-closed or eyes-open stretch."""

from __future__ import annotations

import re

from .condition import CONDITION_LABELS, condition_in_text
from .recording import Mark, Recording

_SPAN = re.compile(r"mark, (\d+\.\d–\d+\.\d s)")


def span_text(notes: list[str]) -> str:
    """Return the ``0.0–4.0 s`` span from a mark note, or an empty string."""
    for note in reversed(notes):
        match = _SPAN.search(note)
        if match:
            return match.group(1)
    return ""


def describe_marks(marks: list[Mark]) -> str:
    """One status line for the marks on a file. Empty when there are none."""
    if not marks:
        return ""
    first = marks[0]
    label = first.label.strip() or "unlabeled"
    text = f"WinEEG mark at {first.onset_s:.1f} s: {label}"
    extra = len(marks) - 1
    if extra == 1:
        text += " (1 more)"
    elif extra > 1:
        text += f" ({extra} more)"
    return text + "."


def split_marked_conditions(recording: Recording) -> dict[str, Recording]:
    """Clip out each condition a mark names.

    A mark with no duration runs until the next mark, or the end of the file.
    A clock stamp or any other label is left on the recording and does not cut it.
    Spans shorter than one second are ignored.
    """
    ordered = sorted(recording.marks, key=lambda mark: (mark.onset_s, mark.label))
    chosen: dict[str, tuple[float, float, str]] = {}
    duration = recording.duration_s
    for index, mark in enumerate(ordered):
        condition = condition_in_text(mark.label)
        if condition is None:
            continue
        start = max(0.0, float(mark.onset_s))
        if start >= duration:
            continue
        if mark.duration_s > 0:
            end = min(duration, start + float(mark.duration_s))
        else:
            later = [
                other.onset_s
                for other in ordered[index + 1 :]
                if other.onset_s > start + 1e-6
            ]
            end = min(duration, later[0] if later else duration)
        length = end - start
        if length < 1.0:
            continue
        previous = chosen.get(condition)
        if previous is None or length > previous[1]:
            chosen[condition] = (start, length, mark.label.strip())

    return {
        condition: _clip_span(recording, start, length, condition, label)
        for condition, (start, length, label) in chosen.items()
    }


def _clip_span(
    recording: Recording,
    start: float,
    length: float,
    condition: str,
    label: str,
) -> Recording:
    i0 = int(round(start * recording.sfreq))
    i1 = i0 + max(1, int(round(length * recording.sfreq)))
    i0 = min(max(i0, 0), recording.n_samples)
    i1 = min(max(i1, i0), recording.n_samples)
    if i1 <= i0:
        raise ValueError("That mark is past the end of the recording.")
    t0 = i0 / recording.sfreq
    t1 = i1 / recording.sfreq
    name = CONDITION_LABELS.get(condition, condition)
    shown = label or name
    note = f"{name} mark, {t0:.1f}–{t1:.1f} s ({shown})."
    return recording.cropped(i0, i1, note)
