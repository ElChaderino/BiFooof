"""In-memory EEG recording in microvolts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Mark:
    """One annotation, timed from the start of this recording."""

    onset_s: float
    label: str
    duration_s: float = 0.0


@dataclass
class Recording:
    """One continuous recording.

    ``data`` is shaped ``(n_channels, n_samples)`` and stored in microvolts.
    Channel names use modern 10-20 labels when the source names can be
    recognized. Legacy temporal names are stored in their modern form
    (T3/T4/T5/T6 become T7/T8/P7/P8).
    """

    data: np.ndarray
    ch_names: list[str]
    sfreq: float
    source_path: str
    source_format: str
    notes: list[str] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    valid: np.ndarray | None = None
    bad_channels: list[str] = field(default_factory=list)
    rejected_spans: list[tuple[float, float]] = field(default_factory=list)
    # Ear or mastoid channels kept for a linked-ear reference. They are not fitted.
    reference_channels: dict[str, np.ndarray] = field(default_factory=dict)
    dropped_by_amplitude: list[str] = field(default_factory=list)
    # Sample zero's position in the original source, preserved through crops.
    source_start_s: float = 0.0
    reference_applied: str = "stored"
    import_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("Recording data must be a 2-D channels-by-samples array.")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(
                f"{self.data.shape[0]} data rows but {len(self.ch_names)} channel names."
            )
        if not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError("Sampling rate must be positive.")
        if not np.isfinite(self.source_start_s):
            raise ValueError("The source start must be finite.")
        if self.valid is not None:
            self.valid = np.ascontiguousarray(self.valid, dtype=bool)
            if self.valid.shape != self.data.shape:
                raise ValueError("The validity mask must have the same shape as the data.")
        cleaned_refs: dict[str, np.ndarray] = {}
        for name, values in self.reference_channels.items():
            row = np.ascontiguousarray(values, dtype=np.float64).ravel()
            if row.size != self.n_samples:
                raise ValueError(f"Reference channel {name} does not match the recording length.")
            cleaned_refs[str(name)] = row
        self.reference_channels = cleaned_refs

    def cropped(self, i0: int, i1: int, note: str = "") -> Recording:
        """Samples ``i0:i1``, including ear references. A full span with no note returns this object."""
        i0 = min(max(int(i0), 0), self.n_samples)
        i1 = min(max(int(i1), i0), self.n_samples)
        if i1 <= i0:
            raise ValueError("That time window is past the end of the recording.")
        if i0 == 0 and i1 == self.n_samples and not note:
            return self
        t0 = i0 / self.sfreq
        t1 = i1 / self.sfreq
        kept = [
            Mark(mark.onset_s - t0, mark.label, mark.duration_s)
            for mark in self.marks
            if t0 - 1e-9 <= mark.onset_s < t1 + 1e-9
        ]
        valid, spans = self.slice_exclusions(i0, i1, t0)
        refs = {
            name: np.copy(np.asarray(values[i0:i1], dtype=np.float64))
            for name, values in self.reference_channels.items()
        }
        notes = list(self.notes)
        if note:
            notes.append(note)
        else:
            notes.append(f"Fit window {t0:.1f}–{t1:.1f} s.")
        return Recording(
            data=np.copy(np.asarray(self.data[:, i0:i1], dtype=np.float64)),
            ch_names=list(self.ch_names),
            sfreq=self.sfreq,
            source_path=self.source_path,
            source_format=self.source_format,
            notes=notes,
            marks=kept,
            valid=valid,
            bad_channels=list(self.bad_channels),
            rejected_spans=spans,
            reference_channels=refs,
            dropped_by_amplitude=list(self.dropped_by_amplitude),
            source_start_s=self.source_start_s + t0,
            reference_applied=self.reference_applied,
            import_info=dict(self.import_info),
        )

    def mask(self, index: int) -> np.ndarray:
        """True for samples that may enter a fit on this channel."""
        if self.valid is None:
            return np.ones(self.n_samples, dtype=bool)
        return np.asarray(self.valid[index], dtype=bool)

    def analysis_channel(self, name: str) -> bool:
        """Voltage EEG channels eligible for neural analysis and referencing."""
        if name.upper() in {"A1", "A2", "M1", "M2", "LE", "EL", "RE", "ER", "T9", "T10"}:
            return False
        kind = self.import_info.get("analysis_channel_types", {}).get(name, "eeg")
        return kind in {"eeg", "seeg", "ecog", "dbs", "misc"}

    def require_analysis_channel(self, index: int) -> None:
        if not self.analysis_channel(self.ch_names[index]):
            raise ValueError(f"{self.ch_names[index]} is an auxiliary/reference channel, retained for trace review only.")

    def slice_exclusions(self, i0: int, i1: int, t0: float) -> tuple[np.ndarray | None, list[tuple[float, float]]]:
        """Validity mask and rejected spans for samples ``i0:i1``, timed from the slice start."""
        width = max(0, i1 - i0)
        end = t0 + (width / self.sfreq if self.sfreq else 0.0)
        spans = []
        for start, stop in self.rejected_spans:
            if stop <= t0 or start >= end:
                continue
            spans.append((max(0.0, start - t0), min(end - t0, stop - t0)))
        if self.valid is None:
            return None, spans
        piece = self.valid[:, i0:i1]
        if piece.size and bool(piece.all()):
            return None, spans
        return np.copy(piece), spans

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq if self.sfreq else 0.0
