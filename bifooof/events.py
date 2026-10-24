"""Reviewable event tables with explicit state-transition directions."""
from .io.condition import condition_in_text


def event_table(recording, edits=None):
    events = edits if edits is not None else [
        {"onset": mark.onset_s + recording.source_start_s, "duration": mark.duration_s, "trial_type": mark.label}
        for mark in recording.marks
    ]
    result, previous = [], None
    for item in sorted(events, key=lambda row: float(row["onset"])):
        onset, duration = float(item["onset"]), float(item.get("duration", 0))
        if not (-float("inf") < onset < float("inf") and 0 <= duration < float("inf")):
            raise ValueError("Event onsets must be finite; durations must be finite and nonnegative.")
        label = str(item.get("trial_type", "Event"))
        state = item["state"] if "state" in item else condition_in_text(label)
        if state not in (None, "", "ec", "eo"):
            raise ValueError("Event state is ec, eo, or blank.")
        direction = f"{previous}_to_{state}" if previous and state and previous != state else ""
        result.append({"onset": onset, "duration": duration, "trial_type": label,
            "state": state or "", "direction": direction, "channel": str(item.get("channel", "*"))})
        if state:
            previous = state
    return result
