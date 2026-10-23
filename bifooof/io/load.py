"""Load EEG recordings into one microvolt array.

MNE reads EDF/EDF+, BDF, GDF, BrainVision, Neuroscan CNT, EEGLAB, and FIF.
WinEEG ``.eeg`` is read by :mod:`bifooof.io.wineeg`.
BioExplorer ``.bxs`` is converted with ``bxs2edf.exe`` when that tool is present.
CSV and JSON follow the EDF Studio tabular contract.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..analysis.prepare import voltage_scale_note
from .channels import clean_channel_name, is_auxiliary, match_standard_19
from .sniff import sniff
from .marks import describe_marks
from .recording import Mark, Recording
from .wineeg import legacy_physical_channels, parse_legacy_marks, read_legacy_eeg

_BXS_CANDIDATES = [
    Path(os.environ["BIFOOF_BXS2EDF"]) if os.environ.get("BIFOOF_BXS2EDF") else None,
    Path(r"C:\Users\vocal\Desktop\EEG_Paradox_Decoder\bxs2edf.exe"),
    Path(r"C:\Users\vocal\Desktop\Live Paradox\EEG_Paradox_Decoder\bxs2edf.exe"),
    Path(r"E:\EEG_Paradox_Decoder\bxs2edf.exe"),
    Path(r"E:\TQ7\bxs2edf.exe"),
    Path(r"E:\Working\EEG_Paradox_Decoder\bxs2edf.exe"),
]

_TIME_HEADERS = {"time_s", "time", "sec", "seconds"}
_MNE_READERS = {
    "edf": "read_raw_edf",
    "bdf": "read_raw_bdf",
    "gdf": "read_raw_gdf",
    "brainvision": "read_raw_brainvision",
    "eeglab": "read_raw_eeglab",
    "fif": "read_raw_fif",
}


def load_recording(path: str | Path, sfreq: float | None = None) -> Recording:
    """Load ``path`` and return a microvolt :class:`Recording`."""
    source = Path(path)
    kind = sniff(source)
    if kind == "unknown-eeg":
        raise ValueError(
            "This .eeg file is not a WinEEG recording (magic 0x0f0e) and there is no "
            "BrainVision .vhdr beside it."
        )
    if kind == "wineeg":
        return _load_wineeg(source)
    if kind == "bxs":
        return _load_bxs(source)
    if kind == "csv":
        return _load_csv(source, sfreq=sfreq)
    if kind == "json":
        return _load_json(source, sfreq=sfreq)
    if kind == "cnt":
        return _load_cnt(source)
    if kind == "brainvision" and source.suffix.lower() == ".eeg":
        source = source.with_suffix(".vhdr")
    return _load_mne(source, kind, display_path=Path(path))


def needs_sfreq(path: str | Path) -> bool:
    """True when a CSV or JSON file has no sampling rate of its own."""
    source = Path(path)
    try:
        kind = sniff(source)
    except (OSError, ValueError):
        return False
    if kind == "csv":
        _data, inferred, _names = _read_csv_matrix(source)
        return inferred is None
    if kind == "json":
        meta = _read_json(source)
        if "data" in meta and isinstance(meta["data"], list):
            return not _positive(meta.get("sfreq"))
        if meta.get("csv_file"):
            if _positive(meta.get("sfreq")):
                return False
            csv_path = (source.parent / str(meta["csv_file"])).resolve()
            if csv_path.is_file():
                _data, inferred, _names = _read_csv_matrix(
                    csv_path, delimiter=str(meta.get("delimiter") or ",")
                )
                return inferred is None
            return True
    return False


def finalize(
    data: np.ndarray,
    names: list[str],
    sfreq: float,
    source_path: str,
    source_format: str,
    notes: list[str] | None = None,
    valid: np.ndarray | None = None,
    bad_channels: list[str] | None = None,
    rejected_spans: list[tuple[float, float]] | None = None,
) -> Recording:
    """Clean names, drop annotation channels, and prefer the standard 19.

    Samples that are nonfinite, inside a BAD annotation, or on a bad channel
    stay excluded. They are stored as NaN and are not turned into zeros.
    """
    notes = list(notes or [])
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a channels-by-samples array.")
    if array.shape[0] != len(names):
        raise ValueError("Channel name count does not match the data.")
    finite = np.isfinite(array)
    if valid is None:
        accepted = finite
    else:
        accepted = np.asarray(valid, dtype=bool) & finite
        if accepted.shape != array.shape:
            raise ValueError("The validity mask must have the same shape as the data.")
    bad_names = {clean_channel_name(name) for name in (bad_channels or []) if clean_channel_name(name)}
    for index, raw_name in enumerate(names):
        if clean_channel_name(raw_name) in bad_names:
            accepted[index, :] = False

    kept_rows: list[int] = []
    cleaned: list[str] = []
    seen: set[str] = set()
    dropped_aux = 0
    dropped_dup = 0
    for index, raw_name in enumerate(names):
        if is_auxiliary(raw_name):
            dropped_aux += 1
            continue
        label = clean_channel_name(raw_name)
        if not label or is_auxiliary(label):
            dropped_aux += 1
            continue
        if label in seen:
            dropped_dup += 1
            continue
        seen.add(label)
        cleaned.append(label)
        kept_rows.append(index)

    if not kept_rows:
        raise ValueError("No EEG channels remained after dropping annotation and marker channels.")

    array = array[kept_rows]
    accepted = accepted[kept_rows]
    reference_channels: dict[str, np.ndarray] = {}
    for index, name in enumerate(cleaned):
        if name in {"A1", "A2", "M1", "M2"} and name not in reference_channels:
            ear = np.asarray(array[index], dtype=np.float64).copy()
            ear[~accepted[index]] = np.nan
            reference_channels[name] = ear
    matched = match_standard_19(cleaned)
    if matched:
        order = [cleaned.index(name) for name in matched]
        array = array[order]
        accepted = accepted[order]
        cleaned = [cleaned[i] for i in order]
        skipped = len(kept_rows) - len(order)
        if skipped:
            notes.append(f"Kept {len(cleaned)} standard 10-20 channels and set aside {skipped} others.")
    else:
        notes.append("No standard 10-20 channels matched; kept all remaining channels.")
    if dropped_aux:
        notes.append(f"Dropped {dropped_aux} annotation or marker channel(s).")
    if dropped_dup:
        notes.append(f"Dropped {dropped_dup} duplicate channel(s) after renaming.")

    kept_bad = [name for name in cleaned if name in bad_names]
    if kept_bad:
        notes.append("Bad channels left out of the fit: " + ", ".join(kept_bad) + ".")
    excluded = int((~accepted).sum())
    if rejected_spans:
        listed = ", ".join(f"{start:.1f}–{stop:.1f} s" for start, stop in rejected_spans)
        notes.append(f"BAD annotations left out of the fit: {listed}.")
    if excluded:
        notes.append(f"Left {excluded} nonfinite or BAD-marked channel-samples out of the fit.")
        array = array.copy()
        array[~accepted] = np.nan
        stored_valid: np.ndarray | None = accepted
    else:
        stored_valid = None
    if reference_channels:
        kept_ears = ", ".join(name for name in ("A1", "A2", "M1", "M2") if name in reference_channels)
        notes.append(f"Kept {kept_ears} for a linked-ear reference. They are not part of the fit.")
    scale_note = voltage_scale_note(array)
    if scale_note:
        notes.append(scale_note)

    return Recording(
        data=array,
        ch_names=cleaned,
        sfreq=float(sfreq),
        source_path=source_path,
        source_format=source_format,
        notes=notes,
        valid=stored_valid,
        bad_channels=kept_bad,
        rejected_spans=list(rejected_spans or []),
        reference_channels=reference_channels,
    )


def _load_mne(path: Path, kind: str, display_path: Path | None = None) -> Recording:
    import mne

    reader_name = _MNE_READERS[kind]
    reader = getattr(mne.io, reader_name)
    raw = reader(str(path), preload=True, verbose="ERROR")
    shown = display_path or path
    return _recording_from_raw(raw, shown, kind)


def _load_cnt(path: Path) -> Recording:
    import mne

    last: Exception | None = None
    attempts: tuple[dict, ...] = ({}, {"data_format": "int16"}, {"data_format": "int32"})
    for extra in attempts:
        try:
            raw = mne.io.read_raw_cnt(str(path), preload=True, verbose="ERROR", **extra)
            return _recording_from_raw(raw, path, "cnt")
        except Exception as exc:  # CNT integer width is vendor-specific.
            last = exc
    assert last is not None
    raise ValueError(f"Could not read Neuroscan CNT: {last}") from last


def _recording_from_raw(raw, path: Path, kind: str) -> Recording:
    data = np.asarray(raw.get_data(), dtype=np.float64) * 1e6
    sfreq = float(raw.info["sfreq"])
    valid = np.ones(data.shape, dtype=bool)
    spans: list[tuple[float, float]] = []
    annotations = getattr(raw, "annotations", None)
    descriptions = [] if annotations is None else list(annotations.description)
    channel_lists = getattr(annotations, "ch_names", None) if annotations is not None else None
    names = list(raw.ch_names)
    for index, description in enumerate(descriptions):
        if not str(description).upper().startswith("BAD"):
            continue
        onset = float(annotations.onset[index])
        end = onset + float(annotations.duration[index])
        i0 = min(max(0, int(round(onset * sfreq))), data.shape[1])
        i1 = min(max(i0, int(round(end * sfreq))), data.shape[1])
        named: list[str] = []
        if channel_lists is not None and index < len(channel_lists) and channel_lists[index]:
            named = [clean_channel_name(str(name)) for name in channel_lists[index]]
            named = [name for name in named if name]
        if not named:
            valid[:, i0:i1] = False
            spans.append((max(0.0, onset), max(0.0, end)))
            continue
        for row, raw_name in enumerate(names):
            if clean_channel_name(raw_name) in named:
                valid[row, i0:i1] = False
    bads = [str(name) for name in raw.info.get("bads", [])]
    return finalize(
        data, names, sfreq, str(path), kind,
        valid=valid, bad_channels=bads, rejected_spans=spans,
    )


def _load_wineeg(path: Path) -> Recording:
    eeg = read_legacy_eeg(path)
    logical = int(eeg.header.get("logical_frame_count", 0))
    physical = legacy_physical_channels(eeg, logical_only=logical > 0)
    labels = [channel.label for channel in eeg.channels]
    notes: list[str] = []
    sidecar = Path(str(path) + ".conversion.json")
    if sidecar.is_file():
        meta = _read_json(sidecar)
        full = meta.get("full_channel_labels") or []
        restored = 0
        for index, candidate in enumerate(full):
            text = str(candidate).strip()
            if index < len(labels) and text:
                if text != labels[index]:
                    restored += 1
                labels[index] = text
        if restored:
            notes.append(f"Restored {restored} full channel label(s) from the conversion sidecar.")

    rows: list[list[float]] = []
    kept_names: list[str] = []
    for index, (label, values) in enumerate(zip(labels, physical)):
        if label.strip().upper() == "LABEL" or int(eeg.channels[index].kind) == 4:
            continue
        rows.append(values)
        kept_names.append(label)
    if not rows:
        raise ValueError("The WinEEG file has no signal channels.")
    width = min(len(row) for row in rows)
    data = np.asarray([row[:width] for row in rows], dtype=np.float64)
    sfreq = float(eeg.header["sample_rate_hz"])
    marks = [
        Mark(item.onset_ms / 1000.0, item.label, item.duration_ms / 1000.0)
        for item in parse_legacy_marks(eeg)
    ]
    mark_line = describe_marks(marks)
    if mark_line:
        notes.append(mark_line)
    recording = finalize(data, kept_names, sfreq, str(path), "wineeg", notes)
    recording.marks = marks
    return recording


def _find_bxs2edf() -> Path | None:
    for candidate in _BXS_CANDIDATES:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _load_bxs(path: Path) -> Recording:
    exe = _find_bxs2edf()
    if exe is None:
        raise FileNotFoundError(
            "bxs2edf.exe was not found. Place it with the Paradox Decoder or set BIFOOF_BXS2EDF."
        )
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="bifooof_bxs_") as tmp:
        edf_path = Path(tmp) / f"{path.stem}.edf"
        result = subprocess.run(
            [str(exe), str(path.resolve()), str(edf_path.resolve()), "/o"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=flags,
        )
        if result.returncode != 0 or not edf_path.is_file():
            detail = (result.stderr or result.stdout or "bxs2edf.exe failed").strip()
            raise RuntimeError(detail)
        recording = _load_mne(edf_path, "edf", display_path=path)
    recording.source_format = "bxs"
    recording.notes.insert(0, f"Converted BXS with {exe.name}.")
    return recording


def _load_csv(path: Path, sfreq: float | None, unit: str = "uV",
              delimiter: str = ",", channel_names: list[str] | None = None) -> Recording:
    data, inferred, names = _read_csv_matrix(path, delimiter=delimiter)
    if channel_names:
        if len(channel_names) != data.shape[0]:
            raise ValueError("channel_names length does not match the CSV.")
        names = list(channel_names)
    rate = inferred if inferred else (float(sfreq) if sfreq else 0.0)
    if rate <= 0:
        raise ValueError(
            "Sampling rate is required when the CSV has no time_s, time, sec, or seconds column."
        )
    return finalize(_to_microvolts(data, unit), names, rate, str(path), "csv")


def _load_json(path: Path, sfreq: float | None) -> Recording:
    meta = _read_json(path)
    if isinstance(meta.get("data"), list):
        return _load_embedded_json(path, meta, sfreq)
    csv_rel = meta.get("csv_file")
    if not csv_rel:
        raise ValueError("JSON must contain embedded 'data' or a 'csv_file' manifest.")
    csv_path = (path.parent / str(csv_rel)).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    rate = sfreq if sfreq is not None else meta.get("sfreq")
    names = meta.get("channel_names")
    recording = _load_csv(
        csv_path,
        sfreq=float(rate) if _positive(rate) else None,
        unit=str(meta.get("unit") or "uV"),
        delimiter=str(meta.get("delimiter") or ","),
        channel_names=list(names) if names else None,
    )
    recording.source_path = str(path)
    recording.source_format = "json"
    return recording


def _load_embedded_json(path: Path, meta: dict, sfreq: float | None) -> Recording:
    rate = sfreq if sfreq is not None else meta.get("sfreq")
    if not _positive(rate):
        raise ValueError("Embedded JSON requires sfreq.")
    names = [str(name) for name in meta.get("channel_names") or []]
    data = np.asarray(meta["data"], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Embedded JSON 'data' must be a 2-D array.")
    notes: list[str] = []
    if names and data.shape[0] != len(names) and data.shape[1] == len(names):
        data = data.T
        notes.append("Transposed JSON data to channels × samples.")
    if not names:
        names = [f"CH{i + 1}" for i in range(data.shape[0])]
    if len(names) != data.shape[0]:
        raise ValueError("channel_names length does not match the embedded data.")
    unit = str(meta.get("unit") or "uV")
    return finalize(_to_microvolts(data, unit), names, float(rate), str(path), "json", notes)


def _read_csv_matrix(path: Path, delimiter: str = ",") -> tuple[np.ndarray, float | None, list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    if len(rows) < 2:
        raise ValueError("CSV must have a header and at least one data row.")
    header = [cell.strip() for cell in rows[0]]
    body = rows[1:]
    has_time = header[0].lower() in _TIME_HEADERS

    def _floats(cells: list[str]) -> list[float]:
        values = []
        for cell in cells:
            text = cell.strip()
            if text == "":
                raise ValueError("CSV contains an empty numeric cell.")
            values.append(float(text))
        return values

    if has_time:
        names = header[1:]
        times: list[float] = []
        samples: list[list[float]] = []
        for row in body:
            if len(row) != len(header):
                raise ValueError("CSV rows do not match the header length.")
            times.append(float(row[0].strip()))
            samples.append(_floats(row[1:]))
        time_arr = np.asarray(times, dtype=np.float64)
        diffs = np.diff(time_arr)
        if np.any(diffs <= 0):
            raise ValueError("The time column must be strictly increasing.")
        inferred = float(1.0 / np.median(diffs)) if len(time_arr) > 1 else None
        data = np.asarray(samples, dtype=np.float64).T
        return data, inferred, names

    names = header
    samples = []
    for row in body:
        if len(row) != len(header):
            raise ValueError("CSV rows do not match the header length.")
        samples.append(_floats(row))
    data = np.asarray(samples, dtype=np.float64).T
    return data, None, names


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError("JSON recording must be an object.")
    return meta


def _to_microvolts(data: np.ndarray, unit: str) -> np.ndarray:
    token = (unit or "uV").strip().lower()
    if token in {"uv", "µv", "μv", "microvolt", "microvolts"}:
        return np.asarray(data, dtype=np.float64)
    if token in {"v", "volt", "volts"}:
        return np.asarray(data, dtype=np.float64) * 1e6
    raise ValueError(f"Unknown unit {unit!r}. Use uV or V.")


def _positive(value) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False
