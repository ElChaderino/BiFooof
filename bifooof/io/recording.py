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

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("Recording data must be a 2-D channels-by-samples array.")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(
                f"{self.data.shape[0]} data rows but {len(self.ch_names)} channel names."
            )
        if self.sfreq <= 0:
            raise ValueError("Sampling rate must be positive.")
        if self.valid is not None:
            self.valid = np.ascontiguousarray(self.valid, dtype=bool)
            if self.valid.shape != self.data.shape:
                raise ValueError("The validity mask must have the same shape as the data.")

    def mask(self, index: int) -> np.ndarray:
        """True for samples that may enter a fit on this channel."""
        if self.valid is None:
            return np.ones(self.n_samples, dtype=bool)
        return np.asarray(self.valid[index], dtype=bool)

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
        return piece, spans

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq if self.sfreq else 0.0
