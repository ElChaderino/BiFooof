"""Reference choice, amplitude rejection, and the voltage-scale check.

The stored reference and every finite sample stay in place unless a fit
setting says otherwise. A file that is still in volts, or far above the EEG
range, is named and left unchanged.

Average reference subtracts the mean of the fitted channels, the same centering
used to move a linked-ear cross-spectrum onto the average reference. Current
source density is the spherical-spline surface Laplacian. Global scale is that
average reference followed by the Hernández factor: spectra are divided by the
geometric mean of the average-referenced power.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..io.channels import clean_channel_name
from ..io.recording import Recording
from .settings import FitSettings, clip_recording, format_hz
from .segments import accepted_runs

_CSD_FLOOR = 4
_POWER_FLOOR = 1e-12

# Below this, a specparam fit is drawn as a weak result rather than a solid one.
R2_WEAK = 0.6
_EAR_PAIRS = (("A1", "A2"), ("M1", "M2"))
_REFERENCE_ALIASES = {
    "A1": ("A1", "LE", "EL"),
    "A2": ("A2", "RE", "ER"),
    "M1": ("M1", "T9"),
    "M2": ("M2", "T10"),
}
_LEFT_SITES = frozenset({"Fp1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"})
_RIGHT_SITES = frozenset({"Fp2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"})
_MIDLINE_SITES = frozenset({"Fpz", "Fz", "Cz", "Pz", "Oz"})
LONGITUDINAL_BIPOLAR = (
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
)
TRANSVERSE_BIPOLAR = (
    ("F7", "Fp1"), ("Fp1", "Fp2"), ("Fp2", "F8"),
    ("F7", "F3"), ("F3", "Fz"), ("Fz", "F4"), ("F4", "F8"),
    ("T7", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T8"),
    ("P7", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "P8"),
    ("O1", "O2"),
)
CIRCUMFERENTIAL_BIPOLAR = (
    ("Fp1", "Fp2"), ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("O2", "O1"), ("O1", "P7"), ("P7", "T7"), ("T7", "F7"), ("F7", "Fp1"),
)


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
    """Clip the fit window, mask samples past the amplitude limit, then apply the reference.

    Samples rejected by the amplitude limit are excluded before the new reference
    is calculated. Masking keeps the original timestamps, so a spectrum or a
    cycle run does not join the surviving samples into a new adjacency.
    """
    prepared = clip_recording(recording, settings)
    prepared = reject_large_amplitudes(prepared, settings.amplitude_limit_uv)
    result = apply_reference(prepared, settings.reference, settings.amplitude_limit_uv, settings)
    info = dict(result.import_info)
    # Epoch and sliding-window fits pass stored so the reference is not applied
    # again. Keep the request that produced the samples already in hand.
    requested = settings.reference
    if requested in {"", "stored"}:
        previous = result.import_info.get("reference_requested")
        if previous not in {None, "", "stored"}:
            requested = str(previous)
        elif result.reference_applied not in {"", "stored"}:
            requested = result.reference_applied
        else:
            requested = "stored"
    info["reference_requested"] = requested
    if result.import_info is prepared.import_info:
        return _copied(
            result, result.data, "",
            import_info=info,
            reference_applied=result.reference_applied,
        )
    result.import_info = info
    return result


def _without_rejected(recording: Recording) -> Recording:
    """Copy on which a rejected sample is missing, so it cannot enter a reference.

    A review exclusion marks the sample invalid and leaves the voltage in place,
    so the trace can still draw it. The recording passed in is not changed.
    """
    if recording.valid is None:
        return recording
    accepted = np.asarray(recording.valid, dtype=bool)
    if accepted.shape != recording.data.shape or bool(accepted.all()):
        return recording
    data = np.array(recording.data, dtype=np.float64, copy=True)
    data[~accepted] = np.nan
    lookup = {name.casefold(): index for index, name in enumerate(recording.ch_names)}
    refs: dict[str, np.ndarray] = {}
    for name, values in recording.reference_channels.items():
        row = np.array(values, dtype=np.float64, copy=True)
        for alias in _REFERENCE_ALIASES.get(name, (name,)):
            index = lookup.get(alias.casefold())
            if index is None or row.shape != accepted[index].shape:
                continue
            row[~accepted[index]] = np.nan
            break
        refs[name] = row
    return _copied(recording, data, "", reference_channels=refs)


def apply_reference(
    recording: Recording,
    mode: str,
    limit_uv: float = 0.0,
    settings: FitSettings | None = None,
) -> Recording:
    """Return the recording in the requested reference. ``stored`` returns it unchanged."""
    if mode in {"", "stored"}:
        return recording
    recording = _without_rejected(recording)
    if mode == "average":
        eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
        data, note, applied = apply_average_reference(recording.data, recording.ch_names, eligible)
        if not applied:
            return _copied(recording, recording.data, note)
        valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
        valid[eligible] &= np.isfinite(data[eligible])
        return _copied(recording, data, note, valid=valid, reference_applied="average")
    if mode == "left":
        return _apply_one_ear(recording, limit_uv, "left")
    if mode == "right":
        return _apply_one_ear(recording, limit_uv, "right")
    if mode == "linked":
        return _apply_pair(recording, limit_uv, (("A1", "A2"),), "linked", "Linked ears", "Linked-ear")
    if mode == "mastoid":
        return _apply_pair(recording, limit_uv, (("M1", "M2"),), "mastoid", "Linked mastoids", "Linked-mastoid")
    if mode == "vertex":
        return _apply_vertex(recording)
    if mode == "ipsilateral":
        return _apply_lateral(recording, limit_uv, contralateral=False)
    if mode == "contralateral":
        return _apply_lateral(recording, limit_uv, contralateral=True)
    if mode == "csd":
        return _apply_csd(recording)
    if mode == "rest":
        return _apply_rest(recording)
    if mode == "hjorth":
        return _apply_hjorth(recording)
    if mode == "global":
        return _apply_global(recording, settings or FitSettings())
    raise ValueError(
        "The reference is stored, average, linked ears, linked mastoids, left ear, right ear, "
        "vertex, ipsilateral ears, contralateral ears, REST, a local Laplacian, "
        "current source density, or global scale."
    )


def _apply_pair(
    recording: Recording,
    limit_uv: float,
    pairs: tuple[tuple[str, str], ...],
    applied: str,
    label: str,
    phrase: str,
) -> Recording:
    """Subtract one named pair. Ears and mastoids are not substitutes for each other."""
    pair = _ear_pair(recording, limit_uv, label, pairs)
    if isinstance(pair, str):
        return _copied(recording, recording.data, pair)
    left, right, left_name, right_name = pair
    data = np.array(recording.data, dtype=np.float64, copy=True)
    eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
    data[eligible] -= (left + right) / 2.0
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(data)] = False
    return _copied(
        recording, data,
        f"{phrase} reference from {left_name} and {right_name}.",
        valid=valid,
        reference_applied=applied,
        reference_channels={left_name: left, right_name: right},
    )


def apply_average_reference(
    data: np.ndarray,
    names: list[str],
    eligible: np.ndarray,
    contributors: list[str] | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Subtract one fixed mean. Membership does not change from sample to sample.

    The contributors are the analysis channels that have any finite data, unless
    ``contributors`` names the set. A sample is left blank wherever that set is
    incomplete, including when only one channel remains. The stored value is
    not kept inside a result labelled as an average reference.
    """
    values = np.array(data, dtype=np.float64, copy=True)
    allowed = np.asarray(eligible, dtype=bool)
    if contributors is None:
        chosen = [
            index for index, name in enumerate(names)
            if allowed[index] and np.isfinite(values[index]).any()
        ]
    else:
        chosen = []
        for name in contributors:
            if name not in names:
                continue
            index = names.index(name)
            if allowed[index]:
                chosen.append(index)
    if len(chosen) < 2:
        return (
            values,
            "Average reference needs at least two analysis channels. The stored reference was kept.",
            False,
        )
    block = values[chosen]
    complete = np.isfinite(block).all(axis=0)
    mean = np.zeros(values.shape[1], dtype=np.float64)
    if bool(complete.any()):
        mean[complete] = block[:, complete].mean(axis=0)
    rows = np.flatnonzero(allowed)
    if rows.size and bool(complete.any()):
        values[np.ix_(rows, np.flatnonzero(complete))] -= mean[complete]
    if rows.size and bool((~complete).any()):
        values[np.ix_(rows, np.flatnonzero(~complete))] = np.nan
    listed = [names[index] for index in chosen]
    membership = f"{len(listed)} channels" if len(listed) > 6 else _listed(listed)
    note = (
        "Average reference across the channels in the fit. "
        f"The mean stays on {membership} for the whole window. "
        "A sample is left out wherever one of those channels is missing, "
        "including when only one channel remains."
    )
    return values, note, True


def average_contributor_names(recording: Recording) -> list[str]:
    """Analysis channels with any accepted sample. This set is fixed for the recording."""
    names = []
    for index, name in enumerate(recording.ch_names):
        if not recording.analysis_channel(name):
            continue
        row = np.asarray(recording.data[index], dtype=np.float64)
        finite = np.isfinite(row) if recording.valid is None else recording.valid[index] & np.isfinite(row)
        if bool(np.any(finite)):
            names.append(name)
    return names


def csd_contributor_names(recording: Recording) -> list[str]:
    """10-20 analysis channels with any accepted sample. Pages keep this same set."""
    lookup = _montage_lookup()
    return [name for name in average_contributor_names(recording) if name.casefold() in lookup]


def bipolar_derivations(
    data: np.ndarray,
    names: list[str],
    pairs: tuple[tuple[str, str], ...],
    visible: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[int]] | None:
    """Channel differences for pairs that are both present. ``None`` when none are.

    ``visible`` limits the pairs to channels on the page. ``data`` still follows ``names``.
    """
    shown = set(names if visible is None else visible)
    actual = {}
    for name in names:
        actual.setdefault(clean_channel_name(name), name)
    shown_modern = {clean_channel_name(name) for name in shown}
    chosen = [
        (actual[left], actual[right]) for left, right in pairs
        if left in shown_modern and right in shown_modern and left in actual and right in actual
    ]
    if not chosen:
        return None
    rows = [data[names.index(left)] - data[names.index(right)] for left, right in chosen]
    labels = [f"{left}−{right}" for left, right in chosen]
    anchors = [names.index(left) for left, _right in chosen]
    return np.vstack(rows), labels, anchors


def apply_vertex_reference(
    data: np.ndarray,
    names: list[str],
    eligible: np.ndarray | None = None,
    *,
    blank_vertex: bool = False,
) -> tuple[np.ndarray, str, bool]:
    """Subtract Cz from the other analysis channels.

    ``blank_vertex`` replaces Cz with missing samples so a fit does not treat
    the reference line as a channel. Otherwise Cz is the zero line.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(names) or "Cz" not in names:
        return values, "Cz was not in the file. The stored reference was kept.", False
    allowed = np.ones(len(names), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    index = names.index("Cz")
    if not bool(allowed[index]):
        return values, "Cz is not an analysis channel. The stored reference was kept.", False
    vertex = np.array(values[index], dtype=np.float64, copy=True)
    if not np.isfinite(vertex).any():
        return values, "Cz had no finite samples. The stored reference was kept.", False
    others = [row for row, name in enumerate(names) if row != index and bool(allowed[row])]
    if not others:
        return values, "Vertex reference needs a channel besides Cz. The stored reference was kept.", False
    out = np.array(values, dtype=np.float64, copy=True)
    for row in others:
        out[row] = out[row] - vertex
    if blank_vertex:
        out[index] = np.nan
    else:
        out[index] = np.where(np.isfinite(vertex), 0.0, np.nan)
    note = "Vertex reference: each channel minus Cz."
    if blank_vertex:
        note = f"{note} Cz is the reference and was left out of the fit."
    if not np.isfinite(vertex).all():
        note = f"{note} Where Cz was missing, that sample was left out."
    return out, note, True


def _apply_vertex(recording: Recording) -> Recording:
    eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
    data, note, applied = apply_vertex_reference(
        recording.data, recording.ch_names, eligible, blank_vertex=True,
    )
    if not applied:
        return _copied(recording, recording.data, note)
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(data)] = False
    return _copied(recording, data, note, valid=valid, reference_applied="vertex")


def _apply_lateral(recording: Recording, limit_uv: float, *, contralateral: bool) -> Recording:
    label = "Contralateral ears" if contralateral else "Ipsilateral ears"
    pair = _ear_pair(recording, limit_uv, label)
    if isinstance(pair, str):
        return _copied(recording, recording.data, pair)
    left, right, left_name, right_name = pair
    data, note, applied = _lateral_data(
        recording.data, recording.ch_names, left, right, left_name, right_name, contralateral,
        np.array([recording.analysis_channel(name) for name in recording.ch_names]),
    )
    if not applied:
        return _copied(recording, recording.data, note)
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(data)] = False
    mode = "contralateral" if contralateral else "ipsilateral"
    return _copied(
        recording, data, note, valid=valid, reference_applied=mode,
        reference_channels={left_name: left, right_name: right},
    )


def _lateral_data(
    data: np.ndarray,
    names: list[str],
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
    contralateral: bool,
    eligible: np.ndarray,
) -> tuple[np.ndarray, str, bool]:
    label = "Contralateral" if contralateral else "Ipsilateral"
    near_left = right if contralateral else left
    near_right = left if contralateral else right
    near_left_name = right_name if contralateral else left_name
    near_right_name = left_name if contralateral else right_name
    out = np.array(data, dtype=np.float64, copy=True)
    used = False
    aside = []
    for index, name in enumerate(names):
        if not bool(eligible[index]):
            continue
        region = _region(name)
        if region == "left":
            out[index] = out[index] - near_left
        elif region == "right":
            out[index] = out[index] - near_right
        elif region == "midline":
            out[index] = out[index] - (left + right) / 2.0
        else:
            aside.append(name)
            out[index] = np.nan
            continue
        used = True
    if not used:
        return (
            np.asarray(data, dtype=np.float64),
            f"No left, right, or midline channel was available for a {label.lower()} reference. "
            "The stored reference was kept.",
            False,
        )
    note = (
        f"{label} reference: left channels minus {near_left_name}, "
        f"right channels minus {near_right_name}, "
        f"and the midline minus the mean of {left_name} and {right_name}."
    )
    if aside:
        note = f"{note} {', '.join(aside)} {'is' if len(aside) == 1 else 'are'} not on a standard side and {'was' if len(aside) == 1 else 'were'} left out."
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        note = f"{note} Where an ear was missing, that sample was left out."
    return out, note, True


def global_scale_factor(power: np.ndarray) -> float | None:
    """Geometric-mean scale of a channels-by-frequencies power matrix.

    This is the factor in the Cuban ``gsf.m`` when every autospectrum value is
    positive: the exponential of the mean log power. A non-positive value is
    left out of that average. ``None`` means there was no positive power.
    """
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return None
    finite_values = values[np.isfinite(values)]
    peak = float(np.max(finite_values)) if finite_values.size else 0.0
    # Empty Welch bins sit many orders below a real peak. Leave them out so
    # they cannot set the scale. A smooth spectrum keeps every bin.
    floor = max(_POWER_FLOOR, peak * 1e-8) if peak > 0 else _POWER_FLOOR
    logs = []
    for column in values.T:
        finite = column[np.isfinite(column)]
        positive = finite[finite > floor]
        if positive.size == 0:
            continue
        logs.extend(float(value) for value in np.log(positive))
    if not logs:
        return None
    # gsf.m divides by every electrode once each bin has any power. That is this
    # mean when every autospectrum value is positive. Empty bins stay out, so a
    # silent frequency cannot shrink the factor.
    factor = float(np.exp(np.mean(logs)))
    if not np.isfinite(factor) or factor <= 0:
        return None
    return factor


def apply_current_source_density(
    data: np.ndarray,
    names: list[str],
    eligible: np.ndarray | None = None,
    contributors: list[str] | None = None,
    details: dict | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Spherical-spline Laplacian. The array is unchanged when it cannot be applied.

    The result is in µV/m² when ``data`` is in µV. ``contributors`` keeps the
    same spatial operator across pages. A sample that is missing on one of
    those channels is left out of every transformed channel. Four channels is
    the minimum the spline will calculate, not a claim that coverage is dense.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(names):
        return values, "Current source density needs one row for each channel name. The stored reference was kept.", False
    allowed = np.ones(len(names), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if allowed.shape != (len(names),):
        return values, "Current source density needs one row for each channel name. The stored reference was kept.", False
    lookup = _montage_lookup()
    positioned = [
        index for index, name in enumerate(names)
        if allowed[index] and name.casefold() in lookup
    ]
    if contributors is None:
        usable = [index for index in positioned if np.isfinite(values[index]).any()]
    else:
        wanted = {name for name in contributors}
        usable = [index for index in positioned if names[index] in wanted]
    if len(positioned) >= _CSD_FLOOR and len(usable) < _CSD_FLOOR:
        return (
            values,
            "This stretch has no finite samples on enough 10-20 channels for current source density. "
            "The stored reference was kept.",
            False,
        )
    if len(usable) < _CSD_FLOOR:
        return (
            values,
            "Current source density needs at least 4 channels on the 10-20 layout. "
            "The stored reference was kept.",
            False,
        )
    used_names = [names[index] for index in usable]
    block = np.array(values[usable], dtype=np.float64, copy=True)
    joint = np.isfinite(block).all(axis=0)
    if not bool(joint.any()):
        return (
            values,
            "Current source density kept one channel set, but those channels do not share a finite sample. "
            "The stored reference was kept.",
            False,
        )
    try:
        matrix, sphere = _laplacian(tuple(used_names))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return (
            values,
            f"Current source density could not be calculated. {exc} The stored reference was kept.",
            False,
        )
    missing = ~joint
    block[:, missing] = 0.0
    transformed = matrix @ block
    transformed[:, missing] = np.nan
    out = np.array(values, dtype=np.float64, copy=True)
    out[usable] = transformed
    off_layout = [
        names[index] for index, name in enumerate(names)
        if allowed[index] and name.casefold() not in lookup
    ]
    empty = [names[index] for index in positioned if index not in usable]
    note = (
        f"Current source density on {_listed(used_names)}. "
        "These values are µV/m², not microvolts."
    )
    if off_layout:
        note = (
            f"{note} {', '.join(off_layout)} "
            f"{'is' if len(off_layout) == 1 else 'are'} not on the 10-20 layout and "
            f"{'was' if len(off_layout) == 1 else 'were'} left out."
        )
    if empty:
        note = (
            f"{note} {', '.join(empty)} had no finite samples and "
            f"{'was' if len(empty) == 1 else 'were'} left out."
        )
    if not bool(allowed.all()):
        note = f"{note} Auxiliary channels were left out."
    if bool(missing.any()):
        note = (
            f"{note} Where one channel was missing, that sample was left out of "
            "every current-source-density channel."
        )
    origin_x, origin_y, origin_z, radius = sphere
    note = (
        f"{note} Sphere radius {radius:.3f} m, centered at "
        f"({origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}) m. "
        f"Every contributing channel is finite together at {int(joint.sum())} samples."
    )
    if len(used_names) < 8:
        note = (
            f"{note} Four channels is only the minimum this spline will calculate. "
            "A set this small is a sparse surface Laplacian."
        )
    if details is not None:
        details.update({
            "csd_channels": list(used_names),
            "csd_sphere_m": [origin_x, origin_y, origin_z, radius],
            "csd_joint_samples": int(joint.sum()),
            "csd_sparse": len(used_names) < 8,
        })
    for index in range(len(names)):
        if index not in usable:
            out[index] = np.nan
    return out, note, True


_REST_MODEL = "template geometry/spherical model"
_REST_SPACING_MM = 15.0
_REST_EXCLUDE_MM = 30.0


def _apply_rest(recording: Recording) -> Recording:
    """Infinity reference from a template spherical forward model.

    A missing montage or a failed forward solution keeps the stored reference.
    The average reference is not put in its place.
    """
    if recording.reference_applied == "rest":
        return _copied(
            recording, recording.data,
            "REST was already applied. It was not applied a second time.",
            reference_applied="rest",
        )
    eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
    lookup = _montage_lookup()
    positioned = [
        index for index, name in enumerate(recording.ch_names)
        if eligible[index] and name.casefold() in lookup and np.isfinite(recording.data[index]).any()
    ]
    if len(positioned) < _CSD_FLOOR:
        return _copied(
            recording, recording.data,
            "REST needs at least 4 channels on the 10-20 layout so a spherical forward model can be built. "
            "The stored reference was kept.",
        )
    used_names = [recording.ch_names[index] for index in positioned]
    try:
        matrix, sphere = _rest_operator(tuple(used_names))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return _copied(
            recording, recording.data,
            "REST could not build a template spherical forward model. "
            f"{exc} The stored reference was kept.",
        )
    block = np.array(recording.data[positioned], dtype=np.float64, copy=True)
    joint = np.isfinite(block).all(axis=0)
    if not bool(joint.any()):
        return _copied(
            recording, recording.data,
            "REST kept one channel set, but those channels do not share a finite sample. "
            "The stored reference was kept.",
        )
    block[:, ~joint] = 0.0
    transformed = matrix @ block
    transformed[:, ~joint] = np.nan
    out = np.array(recording.data, dtype=np.float64, copy=True)
    out[positioned] = transformed
    for index in range(len(recording.ch_names)):
        if index not in positioned:
            out[index] = np.nan
    center = ", ".join(f"{value:.3f}" for value in sphere[:3])
    note = (
        f"REST — {_REST_MODEL}. "
        f"The forward solution uses a four-shell sphere of radius {sphere[3]:.3f} m, "
        f"centered at ({center}) m, and a {_REST_SPACING_MM:g} mm volume source space "
        f"that excludes sources within {_REST_EXCLUDE_MM:g} mm of the center of that source space. "
        f"Current source density was not used. The values stay in microvolts. "
        f"Channels: {_listed(used_names)}."
    )
    if bool((~joint).any()):
        note = f"{note} Where one channel was missing, that sample was left out of every REST channel."
    valid = np.ones(out.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(out)] = False
    result = _copied(recording, out, note, valid=valid, reference_applied="rest")
    result.import_info.update({
        "rest_model": _REST_MODEL,
        "rest_sphere_m": [float(value) for value in sphere],
        "rest_source_spacing_mm": _REST_SPACING_MM,
        "rest_exclude_mm": _REST_EXCLUDE_MM,
        "rest_channels": list(used_names),
    })
    return result


@lru_cache(maxsize=32)
def _rest_operator(names: tuple[str, ...]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Linear REST operator for these 10-20 names, and the fitted sphere in metres."""
    import mne

    info = mne.create_info(list(names), 256.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="raise",
        verbose="error",
    )
    sphere_model = mne.make_sphere_model("auto", "auto", info, verbose="error")
    source = mne.setup_volume_source_space(
        sphere=sphere_model,
        pos=_REST_SPACING_MM,
        exclude=_REST_EXCLUDE_MM,
        sphere_units="m",
        verbose="error",
    )
    forward = mne.make_forward_solution(
        info, trans=None, src=source, bem=sphere_model,
        meg=False, eeg=True, verbose="error",
    )
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)
    forward_names = list(forward["info"]["ch_names"])
    missing = [name for name in names if name not in forward_names]
    if missing or gain.shape[0] != len(forward_names):
        raise ValueError(
            "The spherical forward model did not include "
            + ", ".join(missing or list(names))
            + "."
        )
    order = [forward_names.index(name) for name in names]
    gain = gain[order]
    centered = gain - gain.mean(axis=0, keepdims=True)
    try:
        inverse = np.linalg.pinv(centered, rtol=1e-6)
    except TypeError:
        inverse = np.linalg.pinv(centered, rcond=1e-6)
    transfer = gain @ inverse
    identity = np.eye(len(names))
    average = identity - identity.mean(axis=0, keepdims=True)
    correction = np.mean(transfer @ average, axis=0, keepdims=True)
    matrix = average + correction
    center = np.asarray(sphere_model["r0"], dtype=np.float64).ravel()
    radius = float(sphere_model["layers"][-1]["rad"])
    return (
        np.array(matrix, dtype=np.float64, copy=True),
        (float(center[0]), float(center[1]), float(center[2]), radius),
    )


# Equal-weight neighbors for a Hjorth local Laplacian. No distance scaling.
# A center is used only when every listed neighbor is in the recording.
_HJORTH_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "Fp1": ("Fp2", "F7", "F3"),
    "Fp2": ("Fp1", "F8", "F4"),
    "F7": ("Fp1", "F3", "T7"),
    "F3": ("Fp1", "Fz", "C3", "F7"),
    "Fz": ("F3", "F4", "Cz"),
    "F4": ("Fp2", "Fz", "C4", "F8"),
    "F8": ("Fp2", "F4", "T8"),
    "T7": ("F7", "C3", "P7"),
    "C3": ("F3", "T7", "Cz", "P3"),
    "Cz": ("Fz", "C3", "C4", "Pz"),
    "C4": ("F4", "T8", "Cz", "P4"),
    "T8": ("F8", "C4", "P8"),
    "P7": ("T7", "P3", "O1"),
    "P3": ("C3", "P7", "Pz", "O1"),
    "Pz": ("Cz", "P3", "P4"),
    "P4": ("C4", "P8", "Pz", "O2"),
    "P8": ("T8", "P4", "O2"),
    "O1": ("P3", "P7", "O2"),
    "O2": ("P4", "P8", "O1"),
}


def _apply_hjorth(recording: Recording) -> Recording:
    """Local Laplacian: each site minus the equal mean of its listed neighbors.

    This stays in microvolts. It is not the spherical-spline current source density.
    A missing listed neighbor leaves that center out. A missing sample on a
    neighbor leaves that instant out. The neighbor set does not shrink to
    whoever is finite at that instant.
    """
    if recording.reference_applied == "hjorth":
        return _copied(
            recording, recording.data,
            "The local Laplacian was already applied. It was not applied a second time.",
            reference_applied="hjorth",
        )
    lookup = {name.casefold(): index for index, name in enumerate(recording.ch_names)}
    out = np.array(recording.data, dtype=np.float64, copy=True)
    used: dict[str, list[str]] = {}
    skipped: list[str] = []
    for center, neighbors in _HJORTH_NEIGHBORS.items():
        center_index = lookup.get(center.casefold())
        if center_index is None or not recording.analysis_channel(recording.ch_names[center_index]):
            continue
        present = []
        for neighbor in neighbors:
            index = lookup.get(neighbor.casefold())
            if index is None or not recording.analysis_channel(recording.ch_names[index]):
                present = []
                break
            present.append(index)
        if len(present) != len(neighbors):
            skipped.append(recording.ch_names[center_index])
            out[center_index] = np.nan
            continue
        block = np.vstack([recording.data[index] for index in present])
        missing = ~np.isfinite(block).all(axis=0) | ~np.isfinite(recording.data[center_index])
        mean = np.zeros(block.shape[1], dtype=np.float64)
        if bool((~missing).any()):
            mean[~missing] = block[:, ~missing].mean(axis=0)
        out[center_index] = recording.data[center_index] - mean
        out[center_index, missing] = np.nan
        used[recording.ch_names[center_index]] = [recording.ch_names[index] for index in present]
    if not used:
        return _copied(
            recording, recording.data,
            "The local Laplacian needs a 10-20 channel and every listed neighbor. "
            "The stored reference was kept.",
        )
    for index, name in enumerate(recording.ch_names):
        if recording.analysis_channel(name) and name not in used:
            out[index] = np.nan
    note = (
        "Local Laplacian: each channel minus the equal mean of its listed neighbors. "
        "Weights are equal. There is no distance scaling. The values stay in microvolts. "
        f"Computed on {_listed(list(used))}."
    )
    if skipped:
        note = (
            f"{note} {', '.join(skipped)} "
            f"{'is' if len(skipped) == 1 else 'are'} missing a listed neighbor and "
            f"{'was' if len(skipped) == 1 else 'were'} left out."
        )
    valid = np.ones(out.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(out)] = False
    result = _copied(recording, out, note, valid=valid, reference_applied="hjorth")
    result.import_info["hjorth_neighbors"] = used
    result.import_info["hjorth_weights"] = "equal, no distance scaling"
    return result


def _apply_csd(recording: Recording) -> Recording:
    if recording.reference_applied == "csd":
        return _copied(
            recording, recording.data,
            "Current source density was already applied. It was not applied a second time.",
            reference_applied="csd",
        )
    eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
    details: dict = {}
    data, note, applied = apply_current_source_density(
        recording.data, recording.ch_names, eligible, details=details,
    )
    if not applied:
        return _copied(recording, recording.data, note)
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(data)] = False
    result = _copied(recording, data, note, valid=valid, reference_applied="csd")
    result.import_info.update(details)
    return result


def _apply_global(recording: Recording, settings: FitSettings) -> Recording:
    if recording.reference_applied == "global":
        return _copied(
            recording, recording.data,
            "The global scale factor was already applied. It was not applied a second time.",
            reference_applied="global",
        )
    eligible = [
        index for index, name in enumerate(recording.ch_names)
        if recording.analysis_channel(name) and np.isfinite(recording.data[index]).any()
    ]
    if len(eligible) < 2:
        return _copied(
            recording, recording.data,
            "Global scale needs an average of at least two channels. The stored reference was kept.",
        )
    averaged = apply_reference(recording, "average", settings.amplitude_limit_uv)
    from .spectra import welch_spectra

    rows = np.array(averaged.data[eligible], dtype=np.float64, copy=True)
    valid = None if averaged.valid is None else np.array(averaged.valid[eligible], copy=True)
    try:
        freqs, power, _nperseg = welch_spectra(
            rows, averaged.sfreq, settings.fmin, settings.fmax, settings.segment_s, valid,
        )
    except ValueError as exc:
        return _copied(
            averaged, averaged.data,
            f"The global scale factor needs a spectrum. {exc} The average reference was kept.",
            reference_applied="average",
        )
    factor = global_scale_factor(power)
    if factor is None or factor < 1e-12 or factor > 1e12:
        return _copied(
            averaged, averaged.data,
            "The global scale factor had no positive average-referenced power to use. "
            "The average reference was kept.",
            reference_applied="average",
        )
    scaled = np.array(averaged.data, dtype=np.float64, copy=True)
    scaled[eligible] = scaled[eligible] / np.sqrt(factor)
    low = format_hz(float(freqs[0])) if len(freqs) else format_hz(settings.fmin)
    high = format_hz(float(freqs[-1])) if len(freqs) else format_hz(settings.fmax)
    result = _copied(
        averaged, scaled,
        (
            f"Global scale factor {factor:.6g} from the average-referenced power "
            f"between {low} and {high} Hz. "
            "The factor is the geometric mean of the positive channel-by-frequency Welch values in that band, "
            "not the geometric mean of channel-wise integrated band powers. "
            "A non-positive value, and a bin at or below the power floor, is left out. "
            "Samples were divided by the square root of the factor, so linear spectra and band power are divided by the factor. "
            "In log10 power the offset decreases by the log10 of the factor, and peak height above the aperiodic background does not. "
            "Each recording is scaled on its own, so a uniform amplitude difference between conditions is removed. "
            "The unscaled average-referenced spectrum is this spectrum multiplied by the factor."
        ),
        reference_applied="global",
    )
    result.import_info["global_scale_factor"] = factor
    result.import_info["global_scale_definition"] = (
        "Geometric mean of positive channel-by-frequency Welch power inside the fit band. "
        "Bins at or below max(1e-12, 1e-8 times the peak power) are left out."
    )
    return result


@lru_cache(maxsize=1)
def _montage_lookup() -> dict[str, str]:
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    return {name.casefold(): name for name in montage.ch_names}


@lru_cache(maxsize=64)
def _laplacian(names: tuple[str, ...]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Surface-Laplacian matrix and the fitted sphere, in metres.

    Rows of the matrix follow ``names``. This is the same spherical spline MNE
    applies to an identity. The sphere is ``(x, y, z, radius)`` after the head
    origin is removed.
    """
    import mne
    from mne.bem import fit_sphere_to_headshape
    from mne.channels.interpolation import _calc_g, _calc_h
    from mne.preprocessing._csd import _compute_csd, _prepare_G

    info = mne.create_info(list(names), 256.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="raise",
        verbose="error",
    )
    radius, origin_head, origin_device = fit_sphere_to_headshape(info, verbose="error")
    origin = np.asarray(origin_head, dtype=np.float64) - np.asarray(origin_device, dtype=np.float64)
    pos = np.array([channel["loc"][:3] for channel in info["chs"]], dtype=np.float64)
    if not np.isfinite(pos).all() or np.isclose(pos, 0.0).all(axis=1).any():
        raise ValueError("Zero or infinite position found in chs")
    pos -= origin
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    cos_dist = np.clip(pos @ pos.T, -1.0, 1.0)
    # MNE defaults: stiffness 4, 50 Legendre terms, lambda2 1e-5.
    legendre = _calc_g(cos_dist, stiffness=4, n_legendre_terms=50)
    heights = _calc_h(cos_dist, stiffness=4, n_legendre_terms=50)
    prepared = _prepare_G(np.array(legendre, dtype=np.float64, copy=True), 1e-5)
    matrix = np.array(_compute_csd(prepared, heights, float(radius)), dtype=np.float64, copy=True)
    center = np.asarray(origin, dtype=np.float64)
    return matrix, (float(center[0]), float(center[1]), float(center[2]), float(radius))


def _laplacian_matrix(names: tuple[str, ...]) -> np.ndarray:
    """Surface-Laplacian matrix for these 10-20 names. Rows follow ``names``."""
    return _laplacian(names)[0]


def _listed(names: list[str]) -> str:
    if len(names) > 6:
        return f"{len(names)} channels of the 10-20 layout"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def reject_large_amplitudes(recording: Recording, limit_uv: float) -> Recording:
    """Mask samples that rise more than ±limit above the channel's linear trend.

    The samples stay at their original timestamps. A constant offset or a
    straight drift, which WinEEG files often carry, is not a spike. A channel
    that stays flat beyond the limit is still masked. Zero keeps every sample.
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


def beyond_trend(values: np.ndarray, limit_uv: float) -> np.ndarray:
    """True where a sample leaves a straight trend by more than the limit.

    An offset or a drift that still carries EEG is not marked. A flat line
    beyond the limit, and a single sample beyond it, are marked. This is the
    same rule the fit uses when it rejects amplitude.
    """
    row = np.asarray(values, dtype=np.float64).ravel()
    limit = float(limit_uv)
    if limit <= 0 or row.size == 0:
        return np.zeros(row.shape, dtype=bool)
    valid = np.isfinite(row)
    return _above_trend(row[None, :], valid[None, :], limit)[0]


def _region(name: str) -> str | None:
    """Left, right, or midline. T3–T6 count as the modern sites they rename to."""
    modern = clean_channel_name(name)
    if modern in _LEFT_SITES:
        return "left"
    if modern in _RIGHT_SITES:
        return "right"
    if modern in _MIDLINE_SITES:
        return "midline"
    return None


def _ear_row(recording: Recording, name: str) -> tuple[np.ndarray, str] | None:
    """The ear stored for referencing, or the same name when it is still a data channel.

    The second value is the label on this recording. LE, RE, T9, and T10 count.
    """
    aliases = {"A1": ("A1", "LE", "EL"), "A2": ("A2", "RE", "ER"), "M1": ("M1", "T9"), "M2": ("M2", "T10")}.get(name, (name,))
    for alias in aliases:
        stored = recording.reference_channels.get(alias)
        if stored is not None:
            return stored, alias
        for index, channel in enumerate(recording.ch_names):
            if channel.upper() == alias.upper():
                return recording.data[index], channel
    return None


def _apply_one_ear(recording: Recording, limit_uv: float, side: str) -> Recording:
    """Subtract one ear from every analysis channel. The other ear is not required."""
    label = "Left ear" if side == "left" else "Right ear"
    found = _one_ear(recording, side, limit_uv, label)
    if isinstance(found, str):
        return _copied(recording, recording.data, found)
    ear, ear_name, skipped = found
    data = np.array(recording.data, dtype=np.float64, copy=True)
    eligible = np.array([recording.analysis_channel(name) for name in recording.ch_names])
    if not bool(eligible.any()):
        return _copied(
            recording, recording.data,
            f"{label} needs a channel besides {ear_name}. The stored reference was kept.",
        )
    for index, name in enumerate(recording.ch_names):
        if eligible[index] and name != ear_name:
            data[index] = data[index] - ear
    if ear_name in recording.ch_names:
        data[recording.ch_names.index(ear_name)] = np.where(np.isfinite(ear), 0.0, np.nan)
    note = f"{label} reference from {ear_name}."
    if skipped:
        note = f"{note} {', '.join(skipped)} could not be used."
    if not np.isfinite(ear).all():
        note = f"{note} Where {ear_name} was missing, that sample was left out."
    valid = np.ones(data.shape, dtype=bool) if recording.valid is None else np.array(recording.valid, copy=True)
    valid[~np.isfinite(data)] = False
    mode = "left" if side == "left" else "right"
    return _copied(
        recording, data, note, valid=valid, reference_applied=mode,
        reference_channels={ear_name: ear},
    )


def _one_ear(
    recording: Recording,
    side: str,
    limit_uv: float,
    label: str,
) -> tuple[np.ndarray, str, list[str]] | str:
    names = ("A1",) if side == "left" else ("A2",)
    saw = False
    emptied = False
    wrong_length = False
    skipped: list[str] = []
    for name in names:
        found = _ear_row(recording, name)
        if found is None:
            continue
        row, name = found
        saw = True
        if row.shape[0] != recording.n_samples:
            wrong_length = True
            skipped.append(name)
            continue
        quiet = _quiet(row, limit_uv)
        if not np.isfinite(quiet).any():
            emptied = True
            skipped.append(name)
            continue
        return quiet, name, skipped
    if not saw:
        return f"{label} was not in the file. The stored reference was kept."
    listed = ", ".join(skipped)
    verb = "was" if len(skipped) == 1 else "were"
    if emptied and not wrong_length:
        limit = float(limit_uv)
        if limit > 0:
            return (
                f"{listed} {verb} in the file, but every sample was beyond ±{limit:g} µV of the trend. "
                "The stored reference was kept."
            )
        return f"{listed} {verb} in the file, but {'it has' if len(skipped) == 1 else 'they have'} no finite samples. The stored reference was kept."
    return (
        f"{listed} {verb} in the file, but {'it does' if len(skipped) == 1 else 'they do'} "
        "not match the recording length. The stored reference was kept."
    )


def _ear_pair(
    recording: Recording,
    limit_uv: float = 0.0,
    label: str = "Linked ears",
    pairs: tuple[tuple[str, str], ...] = (("A1", "A2"),),
) -> tuple[np.ndarray, np.ndarray, str, str] | str:
    """Return the requested pair, or a sentence saying why the stored reference stayed.

    ``pairs`` is tried in order. Linked ears pass the earlobes only. Linked
    mastoids pass M1/M2 only. One placement is not a silent stand-in for the other.
    """
    saw_pair = False
    emptied = False
    names = ""
    for left_name, right_name in pairs:
        found_left = _ear_row(recording, left_name)
        found_right = _ear_row(recording, right_name)
        if found_left is None or found_right is None:
            continue
        left, left_name = found_left
        right, right_name = found_right
        saw_pair = True
        names = f"{left_name} and {right_name}"
        if left.shape[0] != recording.n_samples or right.shape[0] != recording.n_samples:
            continue
        left = _quiet(left, limit_uv)
        right = _quiet(right, limit_uv)
        if not np.isfinite(left).any() or not np.isfinite(right).any():
            emptied = True
            continue
        return left, right, left_name, right_name
    if not saw_pair:
        return f"{label} were not in the file. The stored reference was kept."
    if emptied:
        limit = float(limit_uv)
        if limit > 0:
            return (
                f"{names} were in the file, but every sample was beyond ±{limit:g} µV of the trend. "
                "The stored reference was kept."
            )
        return f"{names} were in the file, but they have no finite samples. The stored reference was kept."
    return f"{names} were in the file, but they do not match the recording length. The stored reference was kept."


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
    """Blank ear samples that leave the trend, using the same rule as the fit.

    An offset with EEG on it stays, so linked ears can subtract it. A flat
    line beyond the limit is blanked.
    """
    row = np.array(values, dtype=np.float64, copy=True)
    row[beyond_trend(row, limit_uv)] = np.nan
    return row


def _copied(
    recording: Recording,
    data: np.ndarray,
    note: str,
    valid: np.ndarray | None = None,
    dropped_by_amplitude: list[str] | None = None,
    reference_applied: str | None = None,
    reference_channels: dict[str, np.ndarray] | None = None,
    import_info: dict | None = None,
) -> Recording:
    return Recording(
        data=data,
        ch_names=list(recording.ch_names),
        sfreq=recording.sfreq,
        source_path=recording.source_path,
        source_format=recording.source_format,
        notes=list(recording.notes) if not note else [*recording.notes, note],
        marks=list(recording.marks),
        valid=recording.valid if valid is None else valid,
        bad_channels=list(recording.bad_channels),
        rejected_spans=list(recording.rejected_spans),
        reference_channels=(
            dict(recording.reference_channels) if reference_channels is None else dict(reference_channels)
        ),
        dropped_by_amplitude=list(
            recording.dropped_by_amplitude if dropped_by_amplitude is None else dropped_by_amplitude
        ),
        source_start_s=recording.source_start_s,
        reference_applied=recording.reference_applied if reference_applied is None else reference_applied,
        import_info=dict(recording.import_info) if import_info is None else dict(import_info),
    )


def sample_provenance(
    recording: Recording,
    used: dict[str, list[tuple[int, int]]] | None = None,
    reference_requested: str | None = None,
) -> dict:
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
    requested = (
        reference_requested
        if reference_requested is not None
        else str(recording.import_info.get("reference_requested", recording.reference_applied))
    )
    applied = recording.reference_applied
    fallback = ""
    if requested != applied and recording.notes:
        fallback = recording.notes[-1]
    payload = {
        "source_start_s": recording.source_start_s,
        "reference_requested": requested,
        "reference_applied": applied,
        "reference_fallback": fallback,
        "reference_contributors": (
            list(recording.import_info.get("rest_channels") or [])
            if recording.reference_applied == "rest"
            else reference_contributors if recording.reference_applied in {"average", "global", "csd"}
            else ["Cz"] if recording.reference_applied == "vertex"
            else list(recording.import_info.get("hjorth_neighbors") or [])
            if recording.reference_applied == "hjorth"
            else list(recording.reference_channels) if recording.reference_applied in {
                "linked", "mastoid", "ipsilateral", "contralateral", "left", "right",
            }
            else []
        ),
        "analyzed_window_duration_s": recording.duration_s,
        "invalid_channel_samples": int((~masks).sum()),
        "accepted_duration_s": float(masks.any(axis=0).sum()) / recording.sfreq,
        "accepted_duration_definition": "Time with at least one channel contributing accepted samples.",
        "channels": channels,
    }
    for key in (
        "global_scale_factor", "global_scale_definition",
        "csd_channels", "csd_sphere_m", "csd_joint_samples", "csd_sparse",
        "rest_model", "rest_sphere_m", "rest_source_spacing_mm", "rest_exclude_mm", "rest_channels",
        "hjorth_neighbors", "hjorth_weights",
    ):
        if key in recording.import_info:
            payload[key] = recording.import_info[key]
    return payload
