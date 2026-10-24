"""Reference choice, amplitude rejection, and the voltage-scale check.

The stored reference and every finite sample stay in place unless a fit
setting says otherwise. A file that is still in volts, or far above the EEG
range, is named and left unchanged.
"""

from __future__ import annotations

import numpy as np

from ..io.recording import Recording
from .settings import FitSettings, clip_recording
from .segments import accepted_runs

# Below this, a specparam fit is drawn as a weak result rather than a solid one.
R2_WEAK = 0.6
_EAR_PAIRS = (("A1", "A2"), ("M1", "M2"))


def voltage_scale_note(data: np.ndarray) -> str | None:
    """Warn when the typical size is not EEG in microvolts. The samples are not rescaled."""
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    typical = float(np.median(np.abs(finite)))
    if 0 < typical < 0.05:
        return (
            "Voltage looks like volts rather than microvolts. "
            "Peak height and burst amplitude assume microvolts."
        )
    if typical > 10000.0:
        return (
            "Voltage is far above a normal EEG range. "
            "Peak height and burst amplitude assume microvolts."
        )
    return None


def prepare_recording(recording: Recording, settings: FitSettings) -> Recording:
    """Clip the fit window, drop huge samples, then apply the chosen reference.

    Rejection comes first so a spike cannot pull the average or the linked ears.
    """
    prepared = clip_recording(recording, settings)
    prepared = reject_large_amplitudes(prepared, settings.amplitude_limit_uv)
    return apply_reference(prepared, settings.reference, settings.amplitude_limit_uv)


def apply_reference(recording: Recording, mode: str, limit_uv: float = 0.0) -> Recording:
    """Return the recording in the requested reference. ``stored`` returns it unchanged."""
    if mode in {"", "stored"}:
        return recording
    if mode == "average":
        data = np.array(recording.data, dtype=np.float64, copy=True)
        mask = np.isfinite(data) if recording.valid is None else recording.valid & np.isfinite(data)
        eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
        mask &= eligible[:, None]
        total = np.where(mask, data, 0.0).sum(axis=0)
        count = mask.sum(axis=0)
        mean = np.divide(total, count, out=np.zeros(data.shape[1], dtype=np.float64), where=count >= 2)
        # One valid channel would be subtracted from itself and become zero.
        data[np.ix_(eligible, count >= 2)] -= mean[count >= 2]
        return _copied(
            recording, data,
            "Average reference across the channels in the fit.",
            reference_applied="average",
        )
    if mode == "linked":
        pair = _ear_pair(recording, limit_uv)
        if pair is None:
            return _copied(
                recording, recording.data,
                "Linked ears were not in the file. The stored reference was kept.",
            )
        left, right, left_name, right_name = pair
        data = np.array(recording.data, dtype=np.float64, copy=True)
        eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
        data[eligible] -= (left + right) / 2.0
        return _copied(
            recording, data,
            f"Linked-ear reference from {left_name} and {right_name}.",
            reference_applied="linked",
        )
    raise ValueError("The reference is stored, average, or linked ears.")


def reject_large_amplitudes(recording: Recording, limit_uv: float) -> Recording:
    """Drop samples that rise more than ±limit above the channel's linear trend.

    A constant offset or a straight drift, which WinEEG files often carry, is
    not a spike. A channel that stays flat beyond the limit is still dropped.
    Zero keeps every sample.
    """
    limit = float(limit_uv)
    if limit <= 0:
        return recording
    data = np.array(recording.data, dtype=np.float64, copy=True)
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    loud = _above_trend(data, valid, limit)
    if not bool(loud.any()):
        return recording
    valid[loud] = False
    data[loud] = np.nan
    emptied = [
        name for index, name in enumerate(recording.ch_names)
        if bool(loud[index].any()) and not bool(valid[index].any())
    ]
    note = f"Left out samples beyond ±{limit:g} µV."
    if emptied:
        note = f"{note} No accepted samples remained on {', '.join(emptied)}."
    dropped = list(recording.dropped_by_amplitude)
    for name in emptied:
        if name not in dropped:
            dropped.append(name)
    return _copied(recording, data, note, valid=valid, dropped_by_amplitude=dropped)


def peak_at_width_floor(bandwidth: float, width_low: float) -> bool:
    """True when a peak is sitting on the narrowest width specparam is allowed to use.

    Reported width is the two-sided bandwidth, the same unit as the width control.
    """
    if not np.isfinite(bandwidth) or width_low <= 0:
        return False
    return float(bandwidth) <= float(width_low) * 1.01 + 1e-9


def _ear_pair(
    recording: Recording,
    limit_uv: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    refs = recording.reference_channels
    for left_name, right_name in _EAR_PAIRS:
        left = refs.get(left_name)
        right = refs.get(right_name)
        if left is None or right is None:
            continue
        if left.shape[0] != recording.n_samples or right.shape[0] != recording.n_samples:
            continue
        left = _quiet(left, limit_uv)
        right = _quiet(right, limit_uv)
        if not np.isfinite(left).any() or not np.isfinite(right).any():
            continue
        return left, right, left_name, right_name
    return None


def _above_trend(data: np.ndarray, valid: np.ndarray, limit: float) -> np.ndarray:
    """Mark samples whose residual from a straight line exceeds the limit.

    The line is fit without the samples that already sit far from the median,
    so one spike cannot tilt it. When most of the channel is far from the
    median, the shape is a drift and the line uses every finite sample.
    """
    loud = np.zeros(data.shape, dtype=bool)
    for index in range(data.shape[0]):
        finite = np.isfinite(data[index]) & valid[index]
        count = int(finite.sum())
        if count == 0:
            continue
        values = data[index, finite]
        if count < 2:
            loud[index, finite] = np.abs(values) > limit
            continue
        position = np.flatnonzero(finite).astype(np.float64)
        center = float(np.median(values))
        far = np.abs(values - center) > limit
        if int((~far).sum()) >= 2 and float(far.mean()) < 0.5:
            slope, intercept = np.polyfit(position[~far], values[~far], 1)
        else:
            slope, intercept = np.polyfit(position, values, 1)
        residual = values - (slope * position + intercept)
        # A flat line beyond the limit has no EEG-sized residual to preserve.
        if float(np.median(np.abs(values))) > limit and float(np.median(np.abs(residual))) <= 1.0:
            loud[index, finite] = True
        else:
            loud[index, finite] = np.abs(residual) > limit
    return loud


def _quiet(values: np.ndarray, limit_uv: float) -> np.ndarray:
    row = np.array(values, dtype=np.float64, copy=True)
    limit = float(limit_uv)
    if limit > 0:
        row[np.isfinite(row) & (np.abs(row) > limit)] = np.nan
    return row


def _copied(
    recording: Recording,
    data: np.ndarray,
    note: str,
    valid: np.ndarray | None = None,
    dropped_by_amplitude: list[str] | None = None,
    reference_applied: str | None = None,
) -> Recording:
    return Recording(
        data=data,
        ch_names=list(recording.ch_names),
        sfreq=recording.sfreq,
        source_path=recording.source_path,
        source_format=recording.source_format,
        notes=[*recording.notes, note],
        marks=list(recording.marks),
        valid=recording.valid if valid is None else valid,
        bad_channels=list(recording.bad_channels),
        rejected_spans=list(recording.rejected_spans),
        reference_channels=dict(recording.reference_channels),
        dropped_by_amplitude=list(
            recording.dropped_by_amplitude if dropped_by_amplitude is None else dropped_by_amplitude
        ),
        source_start_s=recording.source_start_s,
        reference_applied=recording.reference_applied if reference_applied is None else reference_applied,
        import_info=dict(recording.import_info),
    )


def sample_provenance(recording: Recording, used: dict[str, list[tuple[int, int]]] | None = None) -> dict:
    """Counts and source-clock spans of samples contributing to an analysis.

    ``used`` can further restrict prepared samples to runs/epochs actually fit.
    It is omitted only when describing preparation without a completed fit.
    """
    masks = np.isfinite(recording.data)
    if recording.valid is not None:
        masks &= recording.valid
    reference_contributors = [name for index, name in enumerate(recording.ch_names) if recording.analysis_channel(name) and masks[index].any()]
    if used is not None:
        allowed = np.zeros(masks.shape, dtype=bool)
        for index, name in enumerate(recording.ch_names):
            for start, stop in used.get(name, []):
                allowed[index, max(0, start):min(recording.n_samples, stop)] = True
        masks &= allowed
    channels = {}
    for index, name in enumerate(recording.ch_names):
        mask = masks[index]
        channels[name] = {
            "accepted_samples": int(mask.sum()),
            "accepted_duration_s": float(mask.sum()) / recording.sfreq,
            "accepted_spans_source_s": [
                [recording.source_start_s + start / recording.sfreq,
                 recording.source_start_s + stop / recording.sfreq]
                for start, stop in accepted_runs(mask)
            ],
            "excluded_spans_source_s": [
                [recording.source_start_s + start / recording.sfreq,
                 recording.source_start_s + stop / recording.sfreq]
                for start, stop in accepted_runs(~mask)
            ],
        }
    return {
        "source_start_s": recording.source_start_s,
        "reference_applied": recording.reference_applied,
        "reference_contributors": reference_contributors if recording.reference_applied == "average" else list(recording.reference_channels) if recording.reference_applied == "linked" else [],
        "analyzed_window_duration_s": recording.duration_s,
        "invalid_channel_samples": int((~masks).sum()),
        "accepted_duration_s": float(masks.any(axis=0).sum()) / recording.sfreq,
        "accepted_duration_definition": "Time with at least one channel contributing accepted samples.",
        "channels": channels,
    }
