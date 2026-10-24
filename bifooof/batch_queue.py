"""Sequential batch jobs with isolated destinations and retryable item state."""
import csv
import hashlib
from pathlib import Path
from uuid import uuid4

from .batch import run_file
from .io.load import load_recording
from .jobs import Cancelled, checkpoint
from .project import atomic_json, read_json


def run_queue(items, settings, destination, channels=None, notify=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results, aggregate = [], []
    previous = destination / "batch-results.json"
    if previous.exists():
        saved = read_json(previous)
        if saved.get("version") != 1 or not isinstance(saved.get("items"), list):
            raise ValueError("Unsupported existing batch results. Choose a different output folder.")
        results = saved["items"]
    aggregate_path = destination / "batch-results.csv"
    if aggregate_path.exists():
        with aggregate_path.open(encoding="utf-8", newline="") as stream:
            aggregate = list(csv.DictReader(stream))
    for position, item in enumerate(items):
        checkpoint(position, len(items), f"Batch recording {position + 1} of {len(items)}")
        path = Path(item["path"])
        identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        folder = destination / f"{path.stem}-{identity}-{uuid4().hex[:8]}"
        if notify:
            notify(item["index"], "Running", str(folder))
        try:
            selected = None
            if channels:
                recording = load_recording(path)
                selected = [name for name in channels if name in recording.ch_names]
                if not selected:
                    raise ValueError("No members of the selected channel group are present.")
            run_file(path, folder, condition=item["condition"], channel=None if not selected else selected[0],
                band=f"{settings.alpha_low:g}-{settings.alpha_high:g}", plots=False, settings=settings,
                cycle_channels=selected)
            with (folder / "spectrum.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    aggregate.append({"output": str(folder), "batch_source": str(path), **row})
            result = {**item, "status": "Complete", "output": str(folder), "profile_settings": vars(settings)}
        except Cancelled:
            if notify:
                notify(item["index"], "Cancelled", str(folder))
            raise
        except Exception as exc:
            result = {**item, "status": "Failed", "error": str(exc), "output": str(folder)}
        # Replace this queue item's latest attempt while retaining the other outputs.
        key = lambda entry: (entry.get("index"), str(Path(entry["path"]).resolve()), entry["condition"])
        old_outputs = {entry.get("output") for entry in results if key(entry) == key(item)}
        aggregate = [row for row in aggregate if row.get("output") not in old_outputs]
        results = [entry for entry in results if key(entry) != key(item)] + [result]
        if notify:
            notify(item["index"], result["status"] + (": " + result["error"] if "error" in result else ""), str(folder))
        atomic_json(destination / "batch-results.json", {"version": 1, "items": results}, backup=False)
        from .export import write_csv_stream
        write_csv_stream(destination / "batch-results.csv", iter(aggregate))
    return results
