"""Stage a complete export and replace only BiFooof-owned output files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from .jobs import checkpoint


_MANIFEST = "bifooof-manifest.json"
_OWNED = {
    "spectrum.csv", "spectrum.png", "peaks.csv", "metadata.json",
    "cycles.csv", "cycles.png", "cycle_summary.csv", "burst_episodes.csv",
    "comparison.csv", "comparison.png", "cycle_comparison.csv",
    "traces.png", "findings.txt", "time.csv", "time.png", _MANIFEST,
    "report.html", "report.pdf", "review.json", "events.tsv", "events.json", "channels.tsv", "measurement_dictionary.json", "quality.csv", "regions.csv", "epochs.csv",
}
for _condition in ("ec", "eo"):
    _OWNED.update(f"{_condition}_{suffix}" for suffix in (
        "spectrum.csv", "spectrum.png", "peaks.csv", "metadata.json",
        "cycles.csv", "cycle_summary.csv", "burst_episodes.csv", "compare_cycles.csv",
    ))


def save_bundle(folder: str | Path, writer) -> list[Path]:
    """Publish only after all calculations/files succeed; roll back write errors.

    Canonical app output names are reserved. Other files in the folder are kept.
    The manifest is published last, identifying the completed bundle.
    """
    destination = Path(folder).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bifooof-export-", dir=destination.parent) as tmp:
        work = Path(tmp).resolve()
        stage, backup = work / "new", work / "previous"
        stage.mkdir()
        backup.mkdir()
        writer(stage)
        checkpoint(message="Publishing completed export")
        files = sorted(stage.iterdir())
        if any(not path.is_file() or path.is_symlink() or path.name not in _OWNED for path in files):
            raise ValueError("An export contained an unexpected output path.")
        manifest = {
            "format": "bifooof-export", "version": 1, "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": [path.name for path in files],
        }
        (stage / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        files.append(stage / _MANIFEST)
        destination.mkdir(parents=True, exist_ok=True)
        previous, published = [], []
        try:
            for name in sorted(_OWNED):
                old = destination / name
                if old.is_file() or old.is_symlink():
                    os.replace(old, backup / name)
                    previous.append(name)
            for path in files:
                os.replace(path, destination / path.name)
                published.append(path.name)
        except Exception:
            for name in reversed(published):
                (destination / name).unlink()
            for name in previous:
                os.replace(backup / name, destination / name)
            raise
        return [destination / path.name for path in files]
