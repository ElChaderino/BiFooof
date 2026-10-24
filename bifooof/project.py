"""Versioned, atomic projects, reusable profiles, and source verification."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .analysis.settings import FitSettings
from .jobs import checkpoint


def state_directory():
    if os.environ.get("BIFOOOF_STATE_DIR"):
        return Path(os.environ["BIFOOOF_STATE_DIR"])
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return Path(tempfile.gettempdir()) / f"bifooof-test-{os.getpid()}"
    return Path.home() / ".bifooof"


def atomic_json(path, payload, backup=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False)
    descriptor, temporary = tempfile.mkstemp(prefix=".bifooof-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if backup and path.is_file():
            shutil.copy2(path, str(path) + ".previous")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid number: {value}")))


def settings_from(values):
    if not isinstance(values, dict):
        raise ValueError("Settings must be an object.")
    unknown = set(values) - set(FitSettings.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown settings: " + ", ".join(sorted(unknown)))
    settings = FitSettings(**values)
    try:
        settings.validate()
    except (TypeError, OverflowError) as exc:
        raise ValueError("Settings have invalid numeric types.") from exc
    return settings


def source_files(path):
    path = Path(path).resolve()
    files = {"primary": path}
    suffix = path.suffix.lower()
    header = path if suffix == ".vhdr" else path.with_suffix(".vhdr") if suffix == ".eeg" else None
    if header is not None and header.exists():
        if header != path:
            files["header"] = header
        for line in header.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() in {"datafile", "markerfile"}:
                files[key.strip().lower()] = (header.parent / value.strip()).resolve()
    if suffix == ".json":
        manifest = read_json(path)
        if manifest.get("csv_file"):
            files["csv"] = (path.parent / manifest["csv_file"]).resolve()
    if suffix == ".set" and path.with_suffix(".fdt").exists():
        files["fdt"] = path.with_suffix(".fdt")
    if suffix == ".set":
        from scipy.io import loadmat
        try:
            metadata = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["EEG", "data"])
            data_file = getattr(metadata.get("EEG"), "data", metadata.get("data"))
            if isinstance(data_file, str):
                files["fdt"] = (path.parent / data_file).resolve()
        except (ValueError, NotImplementedError, OSError):
            # The reader itself reports unsupported MATLAB encodings.
            pass
    return files


def fingerprint(path):
    found = []
    for role, file in source_files(path).items():
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                checkpoint(message=f"Verifying {file.name}")
                digest.update(block)
        found.append({"role": role, "path": str(file), "size": file.stat().st_size, "sha256": digest.hexdigest()})
    return found


def load_verified_source(path, sfreq=None, channel_policy="standard"):
    from .io.load import load_recording
    files = source_files(path)
    before = {role: (file.stat().st_size, file.stat().st_mtime_ns) for role, file in files.items()}
    identity = fingerprint(path)
    source = load_recording(path, sfreq, channel_policy=channel_policy)
    after = {role: (file.stat().st_size, file.stat().st_mtime_ns) for role, file in files.items()}
    if before != after:
        raise ValueError("The source changed while it was being read. Open it again after writing finishes.")
    source.import_info["source_identity"] = identity
    source.data.flags.writeable = False
    return source


def verify_source(path, expected):
    actual = {item["role"]: item for item in fingerprint(path)}
    if set(actual) != {item["role"] for item in expected}:
        raise ValueError(f"Source companions changed: {Path(path).name}")
    for item in expected:
        other = actual[item["role"]]
        if item["sha256"] != other["sha256"] or item["size"] != other["size"]:
            raise ValueError(f"Source content changed: {other['path']}. Open it as a new recording to start a new review.")


class ReviewState:
    """Review decisions are independent of Qt and the immutable source masks."""
    def __init__(self):
        self.value = {"notes": "", "alias": "", "date": "", "bookmarks": [], "exclusions": {}, "bad_channels": {}, "events": {}, "reviewed": [], "regions": {"Posterior": ["O1", "O2", "Pz"], "Frontal": ["F3", "Fz", "F4"], "Central": ["C3", "Cz", "C4"]}, "band_origin": "Fixed band"}
        self._undo, self._redo = [], []
        self.revision = 0

    def change(self, **updates):
        next_value = {**deepcopy(self.value), **deepcopy(updates)}
        if next_value == self.value:
            return
        self._undo.append(deepcopy(self.value))
        self._undo = self._undo[-100:]
        self._redo.clear()
        self.value = next_value
        self.revision += 1

    def undo(self):
        if self._undo:
            self._redo.append(self.value)
            self.value = self._undo.pop()
            self.revision += 1

    def redo(self):
        if self._redo:
            self._undo.append(self.value)
            self.value = self._redo.pop()
            self.revision += 1

    def restore_history(self, history):
        for name in ("undo", "redo"):
            if not isinstance(history.get(name, []), list):
                raise ValueError("Invalid review history.")
            for value in history.get(name, [])[-100:]:
                validator = ReviewState()
                validator.restore(value)
        self._undo = deepcopy(history.get("undo", [])[-100:])
        self._redo = deepcopy(history.get("redo", [])[-100:])

    def restore(self, values):
        if not isinstance(values, dict):
            raise ValueError("Review state must be an object.")
        self.value.update(deepcopy(values))
        for name in ("notes", "alias", "date", "band_origin"):
            if not isinstance(self.value[name], str):
                raise ValueError(f"Review {name} must be text.")
        for name in ("exclusions", "bad_channels", "regions", "events"):
            if not isinstance(self.value[name], dict):
                raise ValueError(f"Review {name} must be an object.")
        for name in ("bookmarks", "reviewed"):
            if not isinstance(self.value[name], list):
                raise ValueError(f"Review {name} must be a list.")
        if not set(self.value["exclusions"]) <= {"ec", "eo"} or not set(self.value["bad_channels"]) <= {"ec", "eo"}:
            raise ValueError("Unknown review condition assignment.")
        for entries in self.value["exclusions"].values():
            for item in entries:
                start, stop = float(item["start_s"]), float(item["end_s"])
                if not (0 <= start < stop < float("inf")):
                    raise ValueError("Invalid exclusion interval in project.")
        for item in self.value["bookmarks"]:
            if item.get("condition") not in {"ec", "eo"}:
                raise ValueError("A bookmark must belong to ec or eo.")
            if not 0 <= float(item["time_s"]) < float("inf") or not 0 <= float(item.get("duration_s", 0)) < float("inf"):
                raise ValueError("Invalid bookmark in project.")
        for name, members in self.value["regions"].items():
            if not isinstance(name, str) or not isinstance(members, list) or not all(isinstance(value, str) for value in members):
                raise ValueError("Regions must name lists of channels.")
        for events in self.value["events"].values():
            if not isinstance(events, list):
                raise ValueError("Reviewed events must be lists.")
            for item in events:
                if not (-float("inf") < float(item["onset"]) < float("inf") and 0 <= float(item.get("duration", 0)) < float("inf")):
                    raise ValueError("Invalid reviewed event interval.")
        self._undo, self._redo = [], []
        self.revision += 1

    def apply(self, source, condition):
        from .analysis.quality import QualitySpan, mask_spans
        spans = [QualitySpan(**item) for item in self.value["exclusions"].get(condition, [])]
        for name in self.value["bad_channels"].get(condition, []):
            spans.append(QualitySpan(name, 0, source.duration_s, "channel", "confirmed", "Reviewer excluded channel"))
        prepared = mask_spans(source, spans, copy_data=False)
        if spans:
            prepared.bad_channels = list(dict.fromkeys([*source.bad_channels, *self.value["bad_channels"].get(condition, [])]))
        return prepared


def profile(settings, time_controls=None, name="Custom"):
    settings.validate()
    return {"format": "bifooof-profile", "version": 1, "name": str(name), "settings": asdict(settings), "time_controls": dict(time_controls or {})}


def read_profile(path):
    value = read_json(path)
    if value.get("format") != "bifooof-profile" or value.get("version") != 1:
        raise ValueError("Unsupported processing profile.")
    settings_from(value["settings"])
    return value


def make_project(sources, applied, staged, review, ui=None, fingerprints=None, summaries=None):
    applied.validate()
    staged.validate()
    identities = fingerprints if fingerprints is not None else {}
    recordings = {}
    for condition, source in sources.items():
        identity = identities.get(source.source_path) or source.import_info.get("source_identity")
        if identity is None:
            identity = fingerprint(source.source_path)
        recordings[condition] = {"path": str(Path(source.source_path).resolve()), "identity": identity,
            "source_start_s": source.source_start_s, "duration_s": source.duration_s,
            "sfreq": source.sfreq, "channels": source.ch_names,
            "channel_policy": getattr(source, "import_info", {}).get("channel_policy", "standard")}
    return {"format": "bifooof-project", "version": 1, "saved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": recordings, "applied": asdict(applied), "staged": asdict(staged),
        "review": deepcopy(review.value), "review_history": {"undo": deepcopy(review._undo), "redo": deepcopy(review._redo)},
        "ui": dict(ui or {}), "summaries": list(summaries or [])}


def read_project(path):
    value = read_json(path)
    if value.get("format") != "bifooof-project" or value.get("version") != 1:
        raise ValueError("Unsupported BiFooof project version.")
    settings_from(value["applied"])
    settings_from(value.get("staged", value["applied"]))
    state = ReviewState()
    state.restore(value.get("review", {}))
    state.restore_history(value.get("review_history", {}))
    if not isinstance(value.get("sources"), dict) or not value["sources"] or not set(value["sources"]) <= {"ec", "eo"}:
        raise ValueError("Invalid project source assignments.")
    ui = value.get("ui", {})
    if not isinstance(ui, dict):
        raise ValueError("Invalid project view settings.")
    if ui.get("active", "ec") not in {"ec", "eo"} or not isinstance(ui.get("channel", ""), str):
        raise ValueError("Invalid project channel/condition view.")
    for name in ("trace", "time", "time_channel_controls"):
        if not isinstance(ui.get(name, {}), dict):
            raise ValueError("Invalid saved controls.")
        for control in ui.get(name, {}).values():
            if isinstance(control, dict):
                if not isinstance(control.get("text", ""), str) or not isinstance(control.get("index", 0), int):
                    raise ValueError("Invalid saved choice control.")
            elif not isinstance(control, (bool, int, float)) or not -float("inf") < control < float("inf"):
                raise ValueError("Invalid saved numeric control.")
    for name in ("hidden_channels", "time_picks"):
        if not isinstance(ui.get(name, []), list) or not all(isinstance(value, str) for value in ui.get(name, [])):
            raise ValueError("Invalid saved channel selection.")
    selection = ui.get("trace_selection")
    if selection is not None and (not isinstance(selection, (list, tuple)) or len(selection) != 2 or
            not 0 <= float(selection[0]) < float(selection[1]) < float("inf")):
        raise ValueError("Invalid saved trace interval.")
    for name in ("trace_time", "cycles_time", "tab"):
        if not 0 <= float(ui.get(name, 0)) < float("inf"):
            raise ValueError("Invalid saved view position.")
    for entry in value["sources"].values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("identity"), list) or not entry["identity"]:
            raise ValueError("Invalid source identity in project.")
    return value


def load_project_sources(project, replacements=None):
    from .io.load import load_recording
    replacements = replacements or {}
    loaded, originals = {}, {}
    for condition, entry in project["sources"].items():
        path = replacements.get(condition, entry["path"])
        verify_source(path, entry["identity"])
        policy = entry.get("channel_policy", "standard")
        key = (path, policy)
        if key not in originals:
            originals[key] = load_recording(path, sfreq=entry["sfreq"], channel_policy=policy)
            originals[key].import_info["source_identity"] = deepcopy(entry["identity"])
        source = originals[key]
        start, duration = float(entry["source_start_s"]), float(entry["duration_s"])
        if not (0 <= start < float("inf") and 0 < duration < float("inf")):
            raise ValueError("Invalid project source interval.")
        i0, i1 = round(start * source.sfreq), round((start + duration) * source.sfreq)
        if i1 > source.n_samples or source.sfreq != entry["sfreq"] or source.ch_names != entry["channels"]:
            raise ValueError("Source layout does not match the saved project.")
        loaded[condition] = source.cropped(i0, i1)
    continuous = next(iter(originals.values())) if len(originals) == 1 and any(item.marks for item in originals.values()) else None
    return loaded, continuous
