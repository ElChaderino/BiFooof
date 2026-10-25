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

    A mark with no duration runs until the next different condition, or the end
    of the file. A written duration stops there too, so it cannot include the
    next condition. Another label of the same condition does not cut that stretch.
    A clock stamp or any other label is left on the recording and does not cut it.
    Spans shorter than one second are ignored. When two separate spans name the
    same condition, the longer one is kept and the other is named.
    """
    ordered = sorted(recording.marks, key=lambda mark: (mark.onset_s, mark.label))
    found: dict[str, list[tuple[float, float, str, str]]] = {}
    duration = recording.duration_s
    for index, mark in enumerate(ordered):
        condition = condition_in_text(mark.label)
        if condition is None:
            continue
        start = max(0.0, float(mark.onset_s))
        if start >= duration:
            continue
        end = min(duration, start + float(mark.duration_s)) if mark.duration_s > 0 else duration
        cut_note = ""
        for other in ordered[index + 1 :]:
            if other.onset_s <= start + 1e-6 or not _other_condition(other.label, condition):
                continue
            if other.onset_s < end - 1e-6:
                end = other.onset_s
                if mark.duration_s > 0:
                    other_name = CONDITION_LABELS.get(condition_in_text(other.label) or "", "condition")
                    cut_note = (
                        f"The written {float(mark.duration_s):g} s duration was cut "
                        f"at the next {other_name.lower()} mark."
                    )
            break
        end = min(duration, end)
        length = end - start
        if length < 1.0:
            continue
        found.setdefault(condition, []).append((start, length, mark.label.strip(), cut_note))

    clips = {}
    for condition, spans in found.items():
        start, length, label, cut_note = max(spans, key=lambda item: item[1])
        aside = [
            item for item in spans
            if not _covered_by(item[:3], (start, length, label))
        ]
        clips[condition] = _clip_span(recording, start, length, condition, label, aside, cut_note)
    return clips


def _other_condition(label: str, condition: str) -> bool:
    """True when a later mark names a condition and it is not the current one."""
    other = condition_in_text(label)
    return other is not None and other != condition


def _covered_by(span: tuple, kept: tuple) -> bool:
    """True when this span is the kept one, or it sits inside the kept one."""
    start, length, _label = span
    kept_start, kept_length, _kept_label = kept
    if abs(start - kept_start) <= 1e-6 and abs(length - kept_length) <= 1e-6:
        return True
    return kept_start - 1e-6 <= start and start + length <= kept_start + kept_length + 1e-6


def _clip_span(
    recording: Recording,
    start: float,
    length: float,
    condition: str,
    label: str,
    aside: list | None = None,
    cut_note: str = "",
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
    if cut_note:
        note = f"{note} {cut_note}"
    if aside:
        listed = ", ".join(
            f"{item[1]:.1f} s at {item[0]:.1f} s" for item in aside
        )
        note = f"{note} Another {name.lower()} span was left out: {listed}."
    return recording.cropped(i0, i1, note)
