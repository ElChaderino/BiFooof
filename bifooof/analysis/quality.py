"""Stretches that should not be trusted, and a way to leave them out.

Detected spans are the ones the rules are sure about: clipping, a flat line,
a step, and the amplitude limit. Suspected spans are muscle, drift, and mains.
File annotations stay confirmed. Nothing here rewrites the file on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.recording import Recording
from .segments import accepted_runs

_FLAT_UV = 0.5
_STEP_UV = 80.0
_DRIFT_UV = 80.0


@dataclass(frozen=True)
class QualitySpan:
    channel: str
    start_s: float
    end_s: float
    kind: str
    level: str
    note: str


def quality_spans(recording: Recording, limit_uv: float = 500.0) -> list[QualitySpan]:
    """Propose spans for every channel. Confirmed file marks are included."""
    found: list[QualitySpan] = []
    for start, stop in recording.rejected_spans:
        found.append(QualitySpan("", float(start), float(stop), "file mark", "confirmed", "Marked on the file."))
    for index, name in enumerate(recording.ch_names):
        signal = np.asarray(recording.data[index], dtype=np.float64)
        found.extend(_runs(name, recording.sfreq, ~np.isfinite(signal), "missing", "confirmed", "Non-finite samples."))
        if recording.valid is not None:
            found.extend(_runs(
                name, recording.sfreq, ~recording.valid[index] & np.isfinite(signal),
                "file mark", "confirmed", "Left out of the fit.",
            ))
        limit = float(limit_uv)
        if limit > 0:
            loud = np.isfinite(signal) & (np.abs(signal) > limit)
            found.extend(_runs(name, recording.sfreq, loud, "amplitude", "detected", f"Beyond ±{limit:g} µV."))
        found.extend(_flat(name, signal, recording.sfreq))
        found.extend(_steps(name, signal, recording.sfreq))
        found.extend(_windows(name, signal, recording.sfreq))
    return _merge(found)


def mask_spans(recording: Recording, spans: list[QualitySpan], *, copy_data=True) -> Recording:
    """Return a copy with these spans left out of later fits. The file is unchanged."""
    if not spans:
        return recording
    data = np.array(recording.data, dtype=np.float64, copy=True) if copy_data else recording.data
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    rejected = list(recording.rejected_spans)
    for span in spans:
        i0 = max(0, int(round(span.start_s * recording.sfreq)))
        i1 = min(recording.n_samples, int(round(span.end_s * recording.sfreq)))
        if i1 <= i0:
            continue
        if span.channel in ("", "*") or span.channel not in recording.ch_names:
            if span.channel not in ("", "*"):
                continue
            valid[:, i0:i1] = False
            if copy_data:
                data[:, i0:i1] = np.nan
        else:
            index = recording.ch_names.index(span.channel)
            valid[index, i0:i1] = False
            if copy_data:
                data[index, i0:i1] = np.nan
        if span.channel in ("", "*"):
            rejected.append((float(span.start_s), float(span.end_s)))
    return Recording(
        data=data,
        ch_names=list(recording.ch_names),
        sfreq=recording.sfreq,
        source_path=recording.source_path,
        source_format=recording.source_format,
        notes=[*recording.notes, "Left out intervals chosen on the quality timeline."],
        marks=list(recording.marks),
        valid=valid,
        bad_channels=list(recording.bad_channels),
        rejected_spans=rejected,
        reference_channels={name: np.array(values, copy=True) if copy_data else values for name, values in recording.reference_channels.items()},
        dropped_by_amplitude=list(recording.dropped_by_amplitude),
        source_start_s=recording.source_start_s,
        reference_applied=recording.reference_applied,
        import_info=dict(recording.import_info),
    )


def project_exclusions(source: Recording, recordings: list[Recording]) -> Recording:
    """Project masks from cropped condition copies onto their continuous source."""
    spans = []
    for recording in recordings:
        if recording.source_path != source.source_path or recording.sfreq != source.sfreq:
            continue
        offset = recording.source_start_s - source.source_start_s
        for index, name in enumerate(recording.ch_names):
            if name not in source.ch_names:
                continue
            mask = recording.mask(index) & np.isfinite(recording.data[index])
            for start, stop in accepted_runs(~mask):
                spans.append(QualitySpan(
                    name, max(0.0, offset + start / source.sfreq),
                    min(source.duration_s, offset + stop / source.sfreq),
                    "session", "confirmed", "Projected from the condition recording.",
                ))
    return mask_spans(source, spans)


def _flat(name: str, signal: np.ndarray, sfreq: float) -> list[QualitySpan]:
    width = max(4, int(round(0.5 * sfreq)))
    hop = max(1, width // 2)
    flat = np.zeros(signal.size, dtype=bool)
    clip = np.zeros(signal.size, dtype=bool)
    for start in range(0, signal.size - width + 1, hop):
        piece = signal[start:start + width]
        finite = piece[np.isfinite(piece)]
        if finite.size < width // 2:
            continue
        stuck = float(np.nanmax(piece) - np.nanmin(piece)) < 1e-6
        if stuck and abs(float(np.nanmax(piece))) >= 20.0:
            clip[start:start + width] = True
        elif stuck or float(np.std(finite)) < _FLAT_UV:
            flat[start:start + width] = True
    return [
        *_runs(name, sfreq, flat, "flat", "detected", "Flat samples."),
        *_runs(name, sfreq, clip, "clip", "detected", "Repeated extreme value."),
    ]


def _steps(name: str, signal: np.ndarray, sfreq: float) -> list[QualitySpan]:
    finite = np.isfinite(signal)
    diffs = np.diff(signal)
    base = np.abs(diffs[np.isfinite(diffs)])
    if base.size == 0:
        return []
    typical = float(np.median(base))
    limit = max(_STEP_UV, typical * 8.0)
    jumps = np.zeros(signal.size, dtype=bool)
    big = np.isfinite(diffs) & (np.abs(diffs) >= limit)
    jumps[1:] = big
    # Keep the sample on either side of the jump.
    jumps[:-1] |= big
    jumps &= finite
    return _runs(name, sfreq, jumps, "step", "detected", "Abrupt step.")


def _windows(name: str, signal: np.ndarray, sfreq: float) -> list[QualitySpan]:
    width = max(16, int(round(2.0 * sfreq)))
    hop = max(1, width // 2)
    found: list[QualitySpan] = []
    freqs = np.fft.rfftfreq(width, 1.0 / sfreq)
    for start in range(0, max(0, signal.size - width + 1), hop):
        piece = signal[start:start + width]
        if not np.isfinite(piece).all():
            continue
        centered = piece - float(np.mean(piece))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        start_s = start / sfreq
        end_s = (start + width) / sfreq
        slope = abs(float(piece[-1] - piece[0]))
        if slope >= _DRIFT_UV:
            found.append(QualitySpan(name, start_s, end_s, "drift", "suspected", "Slow wander inside this window."))
        if _band_ratio(freqs, spectrum, 30.0, min(45.0, sfreq / 2.0 - 1.0), 4.0, 20.0) >= 4.0:
            found.append(QualitySpan(name, start_s, end_s, "muscle", "suspected", "High-frequency power is high for this window."))
        if _mains(freqs, spectrum):
            found.append(QualitySpan(name, start_s, end_s, "mains", "suspected", "Narrow peak at 50 or 60 Hz."))
    return found


def _band_ratio(freqs, spectrum, high_low, high_high, low_low, low_high) -> float:
    if high_high <= high_low:
        return 0.0
    high = spectrum[(freqs >= high_low) & (freqs <= high_high)]
    low = spectrum[(freqs >= low_low) & (freqs <= low_high)]
    if high.size == 0 or low.size == 0:
        return 0.0
    floor = float(np.median(low))
    if floor <= 0:
        return 0.0
    return float(np.mean(high) / floor)


def _mains(freqs, spectrum) -> bool:
    neighbors = spectrum[(freqs >= 40.0) & (freqs <= 70.0)]
    if neighbors.size < 5:
        return False
    floor = float(np.median(neighbors))
    if floor <= 0:
        return False
    for line in (50.0, 60.0):
        index = int(np.argmin(np.abs(freqs - line)))
        if abs(freqs[index] - line) > 1.5:
            continue
        if spectrum[index] >= floor * 6.0:
            return True
    return False


def _runs(name: str, sfreq: float, mask: np.ndarray, kind: str, level: str, note: str) -> list[QualitySpan]:
    spans = []
    for start, stop in accepted_runs(np.asarray(mask, dtype=bool)):
        if stop <= start:
            continue
        spans.append(QualitySpan(name, start / sfreq, stop / sfreq, kind, level, note))
    return spans


def _merge(spans: list[QualitySpan]) -> list[QualitySpan]:
    """Join spans of the same kind that touch, so the timeline is not a stack of hops."""
    grouped: dict[tuple[str, str, str], list[QualitySpan]] = {}
    for span in spans:
        grouped.setdefault((span.channel, span.kind, span.level), []).append(span)
    merged: list[QualitySpan] = []
    for bundle in grouped.values():
        ordered = sorted(bundle, key=lambda item: item.start_s)
        current = ordered[0]
        for span in ordered[1:]:
            if span.start_s <= current.end_s + 1e-6:
                current = QualitySpan(
                    current.channel, current.start_s, max(current.end_s, span.end_s),
                    current.kind, current.level, current.note,
                )
            else:
                merged.append(current)
                current = span
        merged.append(current)
    merged.sort(key=lambda item: (item.start_s, item.channel, item.kind))
    return merged
