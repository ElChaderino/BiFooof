"""Plain-language reading of a specparam and bycycle fit.

The sentences restate measurements that are already on the other pages.
They do not assign a diagnosis, an arousal label, or a clinical score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..io.channels import STANDARD_19
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from .compare import BAND_CHANGE, ChannelComparison, alpha_peak, compare_spectra, nearest_outside_peak
from .cycles import CycleFit
from .prepare import R2_WEAK, peak_at_width_floor
from .settings import NAMED_BANDS, FitSettings, band_phrase, format_hz, format_seconds
from .specparam_fit import ChannelFit, SpectrumFit, _note_sentences, epochs_for_peak

_POSTERIOR = ("O1", "O2", "Oz", "Pz", "P3", "P4", "P7", "P8", "T5", "T6")
_LIMITS = (
    "These sentences describe the fit.",
    "A steeper or flatter exponent is not an arousal score.",
    "Peak height, band power, and burst shape are not a diagnosis.",
    "Age, vigilance, medication, the recording, artifacts, the reference, and the frequency range all change the numbers.",
    "Sharp waves and abnormal fast activity still need a review of the raw EEG. This page has not seen the trace.",
)
_EPOCHS = (
    "The epochs do not overlap, and each one is fit on its own. "
    "Center, height, and width are the medians of the epochs that found the peak. "
    "An epoch that missed the peak is not entered as a height of zero. "
    "The count of epochs that found it does not scale the height."
)
_READING = (
    "Peak height is the oscillatory peak above the aperiodic background. "
    "The exponent is the slope of that background. "
    "A change in band power can come from either one, and the power alone does not say which."
)
# One quiet line in the Recording section. The file names choose it, so a refit
# does not swap it. Nothing in the window calls it anything but a sentence.
_MARGINALIA = (
    "Berger published the first pages in 1929. The field took a few years to admit the handwriting was real.",
    "Adrian would not agree until he had recorded the rhythm from his own scalp.",
    "Grey Walter named delta, then theta, because alpha and beta were already spoken for.",
    "Jasper's committee wrote down the ten-twenty distances in 1958. A tape measure still does the work.",
    "Kamiya asked people to say when they were in alpha. A fair number learned to answer.",
    "Sterman trained cats on the sensorimotor rhythm. The cats were later in no mood to seize.",
    "Elmer Green called theta a door, and was unspecific about the room.",
    "Barbara Brown wrote the popular book. The laboratory version used fewer adjectives.",
    "Fred and Erna Gibbs drew the atlas by hand. The spikes have not grown more polite.",
    "Dietsch ran a Fourier transform on an EEG in 1932. The computers arrived afterwards and took the credit.",
    "Loomis sat up with sleepers and came back with a set of letters for the night.",
    "Penfield mapped the cortex with the patient awake. The useful part was how carefully he listened.",
    "Lopes da Silva stayed with alpha long enough to stop calling it one thing.",
    "Niedermeyer, shortened without much loss: look at the recording.",
)


@dataclass(frozen=True)
class ReportSection:
    title: str
    lines: tuple[str, ...]
    aside: str = ""


def report_text(sections: list[ReportSection]) -> str:
    """Plain text of a findings report, suitable for a note or a text file."""
    blocks: list[str] = []
    for section in sections:
        blocks.append(section.title)
        blocks.extend(section.lines)
        if section.aside:
            blocks.append("")
            blocks.append(section.aside)
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def compile_report(
    recordings: dict[str, Recording],
    spectra: dict[str, SpectrumFit],
    cycles: dict[str, CycleFit],
    band: tuple[float, float],
    channel: str,
    settings: FitSettings | None = None,
    pair: tuple[str, str] | None = None,
) -> list[ReportSection]:
    """Compile the open fits into short sections.

    ``pair`` is the left and right names from Compare. The default is eyes
    closed and eyes open. Pre/post and custom names use the same two fits.
    """
    if not spectra:
        waiting = "The spectrum fit is running. This page fills in when it finishes." if recordings else (
            "Open a recording. This page fills in when the spectrum fit finishes."
        )
        return [ReportSection("Findings", (waiting, *_LIMITS))]

    chosen = settings or FitSettings()
    low, high = float(band[0]), float(band[1])
    phrase = band_phrase(low, high)
    left, right = _pair_names(pair)
    seen: set[str] = set()
    sections = [
        _recordings(recordings, spectra, chosen, left, right, seen),
        _quality(spectra, (low, high), chosen.peak_width_low, left, right),
        ReportSection("What the numbers are", (_READING,)),
        *_band_sections(spectra, low, high, phrase, channel, chosen.peak_width_low, left, right),
    ]
    if "ec" in spectra and "eo" in spectra:
        sections.append(_comparison(
            spectra["ec"], spectra["eo"], (low, high), phrase, channel, chosen.peak_width_low,
            left, right,
        ))
    rhythm = _rhythm(cycles, (low, high), phrase, channel, left, right, seen)
    if rhythm is not None:
        sections.append(rhythm)
    sections.append(ReportSection("What this page does not say", _LIMITS))
    return sections


def _pair_names(pair: tuple[str, str] | None) -> tuple[str, str]:
    if not pair:
        return "Eyes closed", "Eyes open"
    left = pair[0].strip() or "A"
    right = pair[1].strip() or "B"
    return left, right


def _mention(label: str) -> str:
    """Mid-sentence form. The two eye labels stay lowercase; other names stay as typed."""
    if label.lower() == "eyes closed":
        return "eyes closed"
    if label.lower() == "eyes open":
        return "eyes open"
    return label


def _title_name(label: str) -> str:
    if label.lower() == "eyes closed":
        return "Eyes closed"
    if label.lower() == "eyes open":
        return "Eyes open"
    return label


def _recordings(
    recordings: dict[str, Recording],
    spectra: dict[str, SpectrumFit],
    settings: FitSettings,
    left: str,
    right: str,
    seen: set[str],
) -> ReportSection:
    lines: list[str] = []
    labels = {"ec": left, "eo": right}
    note_texts: list[str] = []
    opened = False
    for key in ("ec", "eo"):
        fit = spectra.get(key)
        recording = recordings.get(key)
        if fit is None and recording is None:
            continue
        if opened:
            lines.append("")
        opened = True
        label = _title_name(labels[key])
        file_notes: list[str] = []
        if recording is None:
            lines.append(f"{label}: spectrum fit is ready.")
        else:
            name = recording.source_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            lines.append(
                f"{label}: {name}, {format_seconds(recording.duration_s)} s, "
                f"{recording.n_channels} channels."
            )
            file_notes.extend(
                note for note in recording.notes
                if "microvolts" in note or "voltage" in note.lower()
            )
        if fit is not None:
            lines.append(
                f"Fit window {format_seconds(fit.t_start)}–{format_seconds(fit.t_end)} s, "
                f"{fit.fmin:g}–{format_hz(fit.fmax)} Hz, {fit.aperiodic_mode} slope."
            )
            if fit.epoch_s and fit.epoch_count:
                lines.append(
                    f"{fit.epoch_count} epochs of {format_seconds(fit.epoch_s)} s, not overlapping."
                )
            if fit.notes:
                file_notes.append(fit.notes)
        note_texts.extend(file_notes)
        lines.extend(_consume_notes(file_notes, seen))
    if any(fit.epoch_s and fit.epoch_count for fit in spectra.values()):
        lines.append(_EPOCHS)
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(_reference_lines(spectra, settings, labels, note_texts))
    if not any(line.strip() for line in lines):
        lines.append("No recording is open.")
    return ReportSection("Recording", tuple(lines), aside=_marginalia(recordings))


def _quality(
    spectra: dict[str, SpectrumFit],
    band: tuple[float, float],
    width_low: float,
    left: str,
    right: str,
) -> ReportSection:
    weak: list[str] = []
    floors: list[str] = []
    dropped: list[str] = []
    labels = {"ec": left, "eo": right}
    for key, fit in spectra.items():
        label = _mention(labels.get(key, CONDITION_LABELS.get(key, key)))
        for item in fit.channels:
            if not item.usable:
                reason = item.reason or "left out of the fit"
                dropped.append(f"{item.name} {label} ({reason})")
                continue
            if item.r_squared is not None and item.r_squared < R2_WEAK:
                weak.append(f"{item.name} {label} R² {item.r_squared:.2f}")
            checked = alpha_peak(item, band) or item.dominant_peak
            if checked is not None and peak_at_width_floor(checked[2], width_low):
                floors.append(f"{item.name} {label}")
    lines = []
    if weak:
        lines.append(
            f"R² below {R2_WEAK:.2f}, so the fit is weak: {_join(weak)}. "
            "A weak fit is a poor basis for reading a change."
        )
    if floors:
        lines.append(
            "A peak is at the narrowest width the fit was allowed to use: "
            f"{_join(floors)}."
        )
    if dropped:
        lines.append(f"Left out of the fit: {_join(dropped)}.")
    if not lines:
        lines.append(
            "No channel was flagged. Nothing was weak, nothing was left out, "
            "and no peak sits at the narrowest width allowed."
        )
    return ReportSection("Check the fit first", tuple(lines))


def _band_sections(
    spectra: dict[str, SpectrumFit],
    low: float,
    high: float,
    phrase: str,
    channel: str,
    width_low: float,
    left: str,
    right: str,
) -> list[ReportSection]:
    sections = []
    labels = {"ec": left, "eo": right}
    for key in ("ec", "eo"):
        fit = spectra.get(key)
        if fit is None:
            continue
        label = _title_name(labels[key])
        lines = [_channel_sentence(fit, low, high, phrase, channel, label, width_low)]
        posterior = [
            item for item in _ordered(fit.channels)
            if item.name in _POSTERIOR and item.name != channel
        ]
        peaked = [item for item in posterior if alpha_peak(item, (low, high)) is not None]
        quiet = [item.name for item in posterior if item.usable and alpha_peak(item, (low, high)) is None]
        if peaked:
            lines.append("Posterior")
            lines.extend(f"{_peak_phrase(item, (low, high))}." for item in peaked)
        if quiet:
            lines.append(f"No posterior peak centered in {phrase}: {_join(quiet)}.")
        others = [
            item.name for item in _ordered(fit.channels)
            if item.usable and item.name not in _POSTERIOR and item.name != channel
            and alpha_peak(item, (low, high)) is not None
        ]
        if others:
            lines.append(f"Also in {phrase}: {_join(others)}.")
        lines.extend(_elsewhere_lines(fit, low, high))
        lines.append(_exponent_sentence(fit, label))
        sections.append(ReportSection(f"{label}, {phrase}", tuple(lines)))
    return sections


def _comparison(
    eyes_closed: SpectrumFit,
    eyes_open: SpectrumFit,
    band: tuple[float, float],
    phrase: str,
    channel: str,
    width_low: float,
    left: str,
    right: str,
) -> ReportSection:
    rows = compare_spectra(eyes_closed, eyes_open, band, width_low)
    by_name = {row.name: row for row in rows}
    spoken_left = _mention(left)
    spoken_right = _mention(right)
    lines = [
        f"{_title_name(left)} set against {_title_name(right)}.",
        (
            f"Peak difference is {spoken_left} fitted peak height minus {spoken_right} fitted peak height, "
            "above the aperiodic background."
        ),
        _band_change(left, right),
    ]
    chosen = by_name.get(channel)
    if chosen is not None:
        lines.append(_pair_sentence(chosen, phrase, lead=True, left=left, right=right))
    posterior = [
        _pair_sentence(row, phrase, lead=False, left=left, right=right)
        for row in rows
        if row.name in _POSTERIOR and row.name != channel
    ]
    posterior = [line for line in posterior if line]
    if posterior:
        lines.append("Posterior")
        lines.extend(posterior)
    both = sum(1 for row in rows if row.alpha_power_ec is not None and row.alpha_power_eo is not None)
    only_left = sum(1 for row in rows if row.alpha_power_ec is not None and row.alpha_power_eo is None)
    only_right = sum(1 for row in rows if row.alpha_power_ec is None and row.alpha_power_eo is not None)
    neither = len(rows) - both - only_left - only_right
    lines.append(
        f"The peak is on both sides for {both} channels. {_title_name(left)} only: {only_left}. "
        f"{_title_name(right)} only: {only_right}. Neither side: {neither}. "
        "A missing peak stays missing, so those channels have no peak difference."
    )
    slopes = [row for row in rows if row.exponent_delta is not None]
    if slopes and chosen is not None and chosen.exponent_delta is not None:
        lines.append(_slope_sentence(chosen, slopes, left, right))
    return ReportSection(
        f"{_title_name(left)} and {_title_name(right)}, {phrase}",
        tuple(line for line in lines if line),
    )


def _band_change(left: str, right: str) -> str:
    """The Welch percent change, named for the pair on screen."""
    if _mention(left) == "eyes closed" and _mention(right) == "eyes open":
        return BAND_CHANGE
    return (
        "Percent change between the two recordings in mean Welch power inside the band: "
        f"({left} − {right}) / {right} × 100. "
        "It is not the band's share of total spectral power. "
        f"It stays blank when {right} power is missing or not positive."
    )


def _rhythm(
    cycles: dict[str, CycleFit],
    band: tuple[float, float],
    phrase: str,
    channel: str,
    left: str,
    right: str,
    seen: set[str],
) -> ReportSection | None:
    lines = []
    labels = {"ec": left, "eo": right}
    for key in ("ec", "eo"):
        fit = cycles.get(key)
        if fit is None or fit.channel != channel:
            continue
        if abs(fit.band[0] - band[0]) > 0.05 or abs(fit.band[1] - band[1]) > 0.05:
            continue
        label = _title_name(labels[key])
        share = ""
        if fit.burst_fraction is not None:
            share = f", {fit.burst_duration_s:.1f} s ({100.0 * fit.burst_fraction:.0f}% of accepted time)"
        lines.append(
            f"{label}, {channel}: {fit.n_cycles} cycles, {fit.n_burst_cycles} burst cycles "
            f"in {fit.n_burst_episodes} episodes{share}."
        )
        shape = _shape(fit.period_ms, fit.rise_decay, fit.peak_trough, fit.volt_amp, fit.unit)
        if shape:
            lines.append(f"All cycles: {shape}.")
        if fit.n_burst_cycles:
            burst_shape = _shape(
                fit.burst_period_ms, fit.burst_rise_decay, fit.burst_peak_trough, fit.burst_volt_amp, fit.unit,
            )
            if burst_shape:
                lines.append(f"Burst cycles: {burst_shape}.")
        if fit.n_burst_episodes >= 2 and fit.burst_fraction is not None and fit.burst_fraction < 0.5:
            lines.append("The bursts are separate runs, and they cover less than half of the accepted time.")
        elif fit.burst_fraction is not None and fit.burst_fraction >= 0.8 and fit.n_burst_cycles:
            lines.append("Burst cycles cover most of the accepted time.")
        if fit.n_cycles and fit.n_burst_cycles == 0 and fit.burst_note:
            lines.append(fit.burst_note)
        lines.extend(_consume_notes([fit.notes] if fit.notes else [], seen))
    if not lines:
        return None
    lines.append(
        "A burst is a run of cycles that cleared the burst rule. "
        "The first and last cycle of a fit are never counted as bursts."
    )
    return ReportSection(f"Rhythm on {channel}, {phrase}", tuple(lines))


def _channel_sentence(
    fit: SpectrumFit,
    low: float,
    high: float,
    phrase: str,
    channel: str,
    label: str,
    width_low: float,
) -> str:
    item = next((entry for entry in fit.channels if entry.name == channel), None)
    if item is None:
        return f"{channel or 'The selected channel'} is not in the {label.lower()} fit."
    if not item.usable:
        return f"{channel}: {item.reason or 'left out of the fit'}."
    peak = alpha_peak(item, (low, high))
    if peak is None:
        nearest = nearest_outside_peak(item, (low, high))
        sentence = f"{channel} has no peak centered in {phrase}."
        if nearest is not None:
            side = "below" if nearest[0] < low else "above"
            sentence += f" The nearest peak is {nearest[0]:.2f} Hz, {nearest[1]:.2f} Hz {side} the band."
        else:
            sentence += " No other peak was found in the fit range."
        return _with_exponent(sentence, item)
    extra = ""
    count = epochs_for_peak(item, peak)
    if count is not None:
        extra = f" It was present in {count} of {item.n_epochs} epochs."
        if count < item.n_epochs:
            extra += " That count is separate from the height, which is not lowered by the epochs that missed it."
    floor = " The peak is at the narrowest allowed width." if peak_at_width_floor(peak[2], width_low) else ""
    return _with_exponent(
        f"{channel}: peak at {peak[0]:.2f} Hz, height {peak[1]:.2f} above the background, "
        f"width {peak[2]:.2f} Hz.{extra}{floor}",
        item,
    )


def _exponent_sentence(fit: SpectrumFit, label: str) -> str:
    usable = [item for item in fit.channels if item.usable and _finite(item.exponent) is not None]
    if not usable:
        return f"{label} has no exponent to report."
    ordered = sorted(usable, key=lambda item: float(item.exponent))
    values = [float(item.exponent) for item in ordered]
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    flat = ordered[0]
    steep = ordered[-1]
    return (
        f"Exponent over {fit.fmin:g}–{format_hz(fit.fmax)} Hz: median {median:.2f} "
        f"across {len(usable)} channels. Flattest {flat.name} {flat.exponent:.2f}, "
        f"steepest {steep.name} {steep.exponent:.2f}."
    )


def _with_exponent(sentence: str, item: ChannelFit) -> str:
    exponent = _finite(item.exponent)
    if exponent is None:
        return sentence
    return f"{sentence} Exponent {exponent:.2f}."


def _elsewhere_lines(fit: SpectrumFit, low: float, high: float) -> list[str]:
    grouped: dict[str, list[tuple[str, float, float]]] = {}
    for item in _ordered(fit.channels):
        if not item.usable:
            continue
        for center, height, _width in item.peaks:
            if low <= center <= high:
                continue
            name = _named_band(center)
            if name is None:
                continue
            grouped.setdefault(name, []).append((item.name, float(center), float(height)))
    if not grouped:
        return []
    lines = [f"Elsewhere in the {fit.fmin:g}–{format_hz(fit.fmax)} Hz fit"]
    for name, _edge_low, _edge_high in NAMED_BANDS:
        found = grouped.get(name)
        if not found:
            continue
        if len(found) <= 4:
            bits = [f"{channel} {center:.2f} Hz, height {height:.2f}" for channel, center, height in found]
            lines.append(f"{name}: {_join(bits)}.")
            continue
        channels = list(dict.fromkeys(channel for channel, _center, _height in found))
        tallest = max(found, key=lambda item: item[2])
        lines.append(
            f"{name} on {_join(channels)}. {len(found)} peaks. "
            f"Tallest {tallest[0]} {tallest[1]:.2f} Hz, height {tallest[2]:.2f}."
        )
    return lines


def _pair_sentence(
    row: ChannelComparison,
    phrase: str,
    lead: bool,
    left: str,
    right: str,
) -> str:
    name = row.name
    peak = row.alpha_reactivity
    spoken_left = _mention(left)
    spoken_right = _mention(right)
    if row.alpha_power_ec is not None and row.alpha_power_eo is not None and peak is not None:
        gap = ""
        if row.peak_center_difference_hz is not None and row.alpha_cf_ec is not None and row.alpha_cf_eo is not None:
            gap = (
                f" Centers {row.alpha_cf_ec:.2f} Hz on {spoken_left} and {row.alpha_cf_eo:.2f} Hz on {spoken_right}, "
                f"{row.peak_center_difference_hz:.2f} Hz apart."
            )
        sentence = (
            f"{name}: fitted peak is {_direction(peak, left, right)} ({peak:+.2f}).{gap}"
        )
        power = _power_clause(row, left, right)
        if power:
            sentence += " " + power
        if lead and peak * (row.band_power_reactivity_percent or 0.0) < 0:
            sentence += " The fitted peak and the Welch band power move in different directions."
        return sentence
    if row.alpha_power_ec is not None or row.alpha_power_eo is not None:
        on_left = row.alpha_power_ec is not None
        side = spoken_left if on_left else spoken_right
        other = spoken_right if on_left else spoken_left
        center = row.alpha_cf_ec if on_left else row.alpha_cf_eo
        nearest = (row.nearest_cf_eo, row.nearest_distance_eo) if on_left else (
            row.nearest_cf_ec, row.nearest_distance_ec,
        )
        sentence = f"{name}: in-band peak only with {side}"
        if center is not None:
            sentence += f", at {center:.2f} Hz"
        sentence += f". {other[:1].upper() + other[1:]} has no peak in {phrase}"
        if nearest[0] is not None and nearest[1] is not None:
            sentence += f"; the nearest peak is {nearest[0]:.2f} Hz, {nearest[1]:.2f} Hz outside"
        sentence += ". The peak difference stays blank."
        power = _power_clause(row, left, right)
        if power:
            sentence += " " + power
        return sentence
    if not lead:
        return ""
    return f"{name}: no peak centered in {phrase} on either side, so there is no peak difference."


def _power_clause(row: ChannelComparison, left: str, right: str) -> str:
    percent = row.band_power_reactivity_percent
    if percent is None or percent != percent or percent in {float("inf"), float("-inf")}:
        return ""
    if abs(percent) < 1.0:
        return "The Welch band change between the two recordings is about zero."
    shown = f"{abs(percent):.1f}".rstrip("0").rstrip(".")
    return (
        f"The Welch band change is {shown}% higher with {_higher_side(percent, left, right)} "
        "than with the other recording."
    )


def _reference_lines(
    spectra: dict[str, SpectrumFit],
    settings: FitSettings,
    labels: dict[str, str],
    note_texts: list[str],
) -> list[str]:
    """Say which reference the fit used. A request the file could not meet stays stored."""
    described = []
    for key in ("ec", "eo"):
        fit = spectra.get(key)
        if fit is None:
            continue
        mode = (fit.provenance or {}).get("reference_applied") or settings.reference
        described.append((key, str(mode), fit))
    if not described:
        return [_applied_reference(settings.reference, settings, [], [])]
    modes = {mode for _key, mode, _fit in described}
    if len(modes) == 1:
        _key, mode, _fit = described[0]
        fits = [fit for _key, _mode, fit in described]
        notes = note_texts if mode == "stored" else []
        return [_applied_reference(mode, settings, notes, _contributors(fits))]
    lines = []
    for key, mode, fit in described:
        notes = note_texts if mode == "stored" else []
        sentence = _applied_reference(mode, settings, notes, _contributors([fit]))
        lines.append(f"{_title_name(labels[key])}: {sentence}")
    return lines


def _contributors(fits: list[SpectrumFit]) -> list[str]:
    names: list[str] = []
    for fit in fits:
        for name in (fit.provenance or {}).get("reference_contributors") or []:
            text = str(name)
            if text not in names:
                names.append(text)
    return names


def _applied_reference(
    mode: str,
    settings: FitSettings,
    note_texts: list[str],
    contributors: list[str],
) -> str:
    if mode == "stored" and settings.reference not in {"", "stored"}:
        reasons = _fallback_reasons(note_texts)
        if reasons:
            return "Reference: as stored in the file. " + " ".join(reasons)
        return "Reference: as stored in the file. The requested reference was not in the file."
    if mode == "average":
        return "Reference: the average of the channels in the fit."
    if mode == "linked" and len(contributors) >= 2:
        return f"Reference: linked ears, {contributors[0]} and {contributors[1]}."
    if mode == "linked":
        return "Reference: linked ears."
    if mode == "mastoid" and len(contributors) >= 2:
        return f"Reference: linked mastoids, {contributors[0]} and {contributors[1]}."
    if mode == "mastoid":
        return "Reference: linked mastoids."
    if mode == "rest":
        named = ", ".join(contributors) if contributors else "the channels on the 10-20 layout"
        return (
            f"Reference: REST, a template spherical head model, on {named}. "
            "The values stay in microvolts. This is an infinity reference, not the average."
        )
    if mode == "hjorth":
        named = ", ".join(contributors) if contributors else "the 10-20 sites whose neighbors were present"
        return (
            f"Reference: local Laplacian on {named}. "
            "Each site minus the equal mean of its listed neighbors. The values stay in microvolts."
        )
    if mode == "left" and len(contributors) == 1:
        return f"Reference: each channel minus {contributors[0]}."
    if mode == "left":
        return "Reference: each channel minus the left ear (A1, LE, or EL)."
    if mode == "right" and len(contributors) == 1:
        return f"Reference: each channel minus {contributors[0]}."
    if mode == "right":
        return "Reference: each channel minus the right ear (A2, RE, or ER)."
    if mode == "vertex":
        return "Reference: each channel minus Cz. Cz itself is left out of the fit."
    if mode == "ipsilateral":
        return (
            "Reference: each side minus its own ear, and the midline minus the mean of the two ears."
        )
    if mode == "contralateral":
        return (
            "Reference: each side minus the opposite ear, and the midline minus the mean of the two ears."
        )
    if mode == "csd":
        return "Reference: current source density. Amplitudes are µV/m², not microvolts."
    if mode == "global":
        return (
            "Reference: the average of the channels in the fit, then the global scale factor. "
            "The spectra are divided by that factor. Each recording is scaled on its own, "
            "so a uniform amplitude difference between conditions is removed."
        )
    return "Reference: as stored in the file."


def _fallback_reasons(note_texts: list[str]) -> list[str]:
    reasons = []
    for text in note_texts:
        sentences = _note_sentences(text)
        for index, sentence in enumerate(sentences):
            following = sentences[index + 1] if index + 1 < len(sentences) else ""
            if "stored reference was kept" in sentence:
                reason = sentence.replace("The stored reference was kept.", "").strip()
            elif following == "The stored reference was kept.":
                reason = sentence
            else:
                continue
            if reason and reason not in reasons:
                reasons.append(reason)
    return reasons


def _consume_notes(texts: list[str], seen: set[str]) -> list[str]:
    kept = []
    for text in texts:
        sentences = _note_sentences(text)
        index = 0
        while index < len(sentences):
            sentence = sentences[index]
            following = sentences[index + 1] if index + 1 < len(sentences) else ""
            folded = (
                "stored reference was kept" in sentence
                or following == "The stored reference was kept."
            )
            if folded or sentence in seen or _hide_note(sentence):
                seen.add(sentence)
                if following == "The stored reference was kept.":
                    seen.add(following)
                    index += 2
                    continue
                index += 1
                continue
            seen.add(sentence)
            kept.append(sentence)
            index += 1
    return kept


def _hide_note(sentence: str) -> bool:
    """Reference announcements are said once, beside the reference line."""
    if "stored reference was kept" in sentence:
        return True
    if sentence in {
        "Average reference across the channels in the fit.",
        "Vertex reference: each channel minus Cz.",
    }:
        return True
    if sentence.startswith((
        "Left ear reference from ",
        "Right ear reference from ",
        "Linked-ear reference from ",
        "Linked-mastoid reference from ",
        "Ipsilateral reference:",
        "Contralateral reference:",
    )):
        return True
    return False


def _marginalia(recordings: dict[str, Recording]) -> str:
    names = []
    for key in ("ec", "eo"):
        recording = recordings.get(key)
        if recording is not None and recording.source_path:
            names.append(recording.source_path)
    if not names:
        return ""
    pick = hashlib.sha256("\0".join(names).encode("utf-8")).digest()[0]
    return _MARGINALIA[pick % len(_MARGINALIA)]


def _slope_sentence(
    chosen: ChannelComparison,
    slopes: list[ChannelComparison],
    left: str,
    right: str,
) -> str:
    delta = float(chosen.exponent_delta)
    ordered = sorted(slopes, key=lambda row: abs(float(row.exponent_delta)), reverse=True)
    deltas = sorted(float(row.exponent_delta) for row in slopes)
    mid = len(deltas) // 2
    if len(deltas) % 2:
        median = deltas[mid]
    else:
        median = 0.5 * (deltas[mid - 1] + deltas[mid])
    largest = ", ".join(
        f"{row.name} {row.exponent_delta:+.2f}" for row in ordered[:3]
    )
    return (
        f"Exponent, {_mention(right)} minus {_mention(left)}, is {delta:+.2f} on {chosen.name}. "
        f"The median change is {median:+.2f}. Largest changes: {largest}. "
        "A change in the exponent is a broadband shift across the fit range."
    )


def _finite(value) -> float | None:
    """A number that can be printed. Missing, nan, and inf stay out of the sentence."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _shape(
    period: float | None,
    rise: float | None,
    trough: float | None,
    voltage: float | None,
    unit: str = "µV",
) -> str:
    parts = []
    period = _finite(period)
    rise = _finite(rise)
    trough = _finite(trough)
    voltage = _finite(voltage)
    if period is not None:
        parts.append(f"period {period:.0f} ms")
    if rise is not None:
        parts.append(f"rise–decay {rise:.2f}")
    if trough is not None:
        parts.append(f"peak–trough {trough:.2f}")
    if voltage is not None:
        parts.append(f"voltage {voltage:.1f} {unit}")
    return ", ".join(parts)


def _peak_phrase(item: ChannelFit, band: tuple[float, float]) -> str:
    peak = alpha_peak(item, band)
    if peak is None:
        return item.name
    return f"{item.name}: {peak[0]:.2f} Hz, height {peak[1]:.2f}"


def _direction(value: float, left: str, right: str) -> str:
    if abs(value) < 0.005:
        return "about the same on both sides"
    return f"higher with {_higher_side(value, left, right)}"


def _higher_side(value: float, left: str, right: str) -> str:
    return _mention(left) if value > 0 else _mention(right)


def _named_band(center: float) -> str | None:
    for name, low, high in NAMED_BANDS:
        if low <= center <= high:
            return name
    return None


def _ordered(channels: list[ChannelFit]) -> list[ChannelFit]:
    rank = {name: index for index, name in enumerate(STANDARD_19)}
    return sorted(channels, key=lambda item: (rank.get(item.name, len(rank)), item.name))


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
