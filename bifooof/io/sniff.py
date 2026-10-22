"""Decide how to read a recording from its magic bytes and extension."""

from __future__ import annotations

from pathlib import Path

# WinEEG / Mitsar proprietary EEG container.
WINEG_MAGIC = 0x0F0E

MNE_KINDS = {
    ".edf": "edf",
    ".bdf": "bdf",
    ".gdf": "gdf",
    ".vhdr": "brainvision",
    ".cnt": "cnt",
    ".set": "eeglab",
    ".fif": "fif",
}

FILE_FILTER = (
    "EEG recordings (*.edf *.bdf *.gdf *.eeg *.vhdr *.cnt *.set *.fif *.bxs *.csv *.json);;"
    "All files (*.*)"
)


def read_prefix(path: Path, nbytes: int = 8) -> bytes:
    with path.open("rb") as handle:
        return handle.read(nbytes)


def sniff(path: str | Path) -> str:
    """Return a format kind for ``path``.

    Kinds: ``wineeg``, ``brainvision``, ``unknown-eeg``, ``edf``, ``bdf``,
    ``gdf``, ``cnt``, ``eeglab``, ``fif``, ``bxs``, ``csv``, ``json``.

    A ``.eeg`` file is WinEEG when it starts with magic ``0x0f0e``. Otherwise
    it is BrainVision only when a sibling ``.vhdr`` is present. Opening a bare
    BrainVision sample file is refused later with that reason.
    """
    source = Path(path)
    if source.is_dir():
        raise ValueError(
            "EGI .mff folders are not opened directly. Export EDF, or open a single-file recording."
        )
    if not source.is_file():
        raise FileNotFoundError(f"File not found: {source}")

    prefix = read_prefix(source)
    if len(prefix) >= 2 and int.from_bytes(prefix[:2], "little") == WINEG_MAGIC:
        return "wineeg"

    ext = source.suffix.lower()
    if ext == ".eeg":
        header = source.with_suffix(".vhdr")
        if header.is_file():
            return "brainvision"
        return "unknown-eeg"

    if ext in MNE_KINDS:
        return MNE_KINDS[ext]
    if ext == ".bxs":
        return "bxs"
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    raise ValueError(f"Unsupported file type: {ext or source.name}")
