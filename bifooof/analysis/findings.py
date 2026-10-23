"""Plain-language reading of a specparam and bycycle fit.

The sentences restate measurements that are already on the other pages.
They do not assign a diagnosis, an arousal label, or a clinical score.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..io.channels import STANDARD_19
from ..io.condition import CONDITION_LABELS
from ..io.recording import Recording
from .compare import BAND_CHANGE, ChannelComparison, alpha_peak, compare_spectra, nearest_outside_peak
from .cycles import CycleFit
from .prepare import R2_WEAK, peak_at_width_floor
from .settings import NAMED_BANDS, FitSettings, band_phrase
from .specparam_fit import EPOCH_PEAKS, ChannelFit, SpectrumFit, epochs_for_peak

_POSTERIOR = ("O1", "O2", "Oz", "Pz", "P3", "P4", "P7", "P8", "T5", "T6")
_LIMITS = (
    "These sentences describe the fit. A steeper or flatter exponent is not an arousal score. "
    "Peak height, band power, and burst shape are not a diagnosis. Age, vigilance, medication, "
    "the recording, artifacts, the reference, and the frequency range all change the numbers. "
    "Sharp waves and abnormal fast activity still need a review of the raw EEG."
)


@dataclass(frozen=True)
class ReportSection:
    title: str
    lines: tuple[str, ...]


def report_text(sections: list[ReportSection]) -> str:
    """Plain text of a findings report, suitable for a note or a text file."""
    blocks: list[str] = []
    for section in sections:
        blocks.append(section.title)
        blocks.extend(section.lines)
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
        return [ReportSection("Findings", (waiting, _LIMITS))]

    chosen = settings or FitSettings()
    low, high = float(band[0]), float(band[1])
    phrase = band_phrase(low, high)
    left, right = _pair_names(pair)
    sections = [
        _recordings(recordings, spectra, chosen, left, right),
        _quality(spectra, (low, high), chosen.peak_width_low, left, right),
        *_band_sections(spectra, low, high, phrase, channel, chosen.peak_width_low, left, right),
    ]
    if "ec" in spectra and "eo" in spectra:
        sections.append(_comparison(
            spectra["ec"], spectra["eo"], (low, high), phrase, channel, chosen.peak_width_low,
            left, right,
        ))
    rhythm = _rhythm(cycles, (low, high), phrase, channel, left, right)
    if rhythm is not None:
        sections.append(rhythm)
    sections.append(ReportSection("What this page does not say", (_LIMITS,)))
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
) -> ReportSection:
    lines = []
    labels = {"ec": left, "eo": right}
    for key in ("ec", "eo"):
        fit = spectra.get(key)
        recording = recordings.get(key)
        if fit is None and recording is None:
            continue
        label = _title_name(labels[key])
        if recording is None:
            lines.append(f"{label}: spectrum fit is ready.")
        else:
            name = recording.source_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            duration = recording.duration_s
            shown = f"{duration:.0f}" if duration >= 10 else f"{duration:.1f}"
            lines.append(f"{label}: {name}, {shown} s, {recording.n_channels} channels.")
            for note in recording.notes:
                if "microvolts" in note or "voltage" in note.lower():
                    lines.append(note)
        if fit is not None:
            lines.append(
                f"{label} fit window {fit.t_start:.0f}–{fit.t_end:.0f} s, "
                f"{fit.fmin:g}–{fit.fmax:.0f} Hz, {fit.aperiodic_mode} slope."
            )
            if fit.epoch_s and fit.epoch_count:
                lines.append(
                    f"{label} uses {fit.epoch_count} non-overlapping epochs of {fit.epoch_s:.0f} s."
                )
            if fit.notes:
                lines.append(fit.notes)
    if any(fit.epoch_s and fit.epoch_count for fit in spectra.values()):
        lines.append(EPOCH_PEAKS)
    lines.append(_reference_sentence(settings))
    if not lines:
        lines.append("No recording is open.")
    return ReportSection("Recording", tuple(lines))


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
            f"R² below {R2_WEAK:.2f}, so the fit is weak: {_sample(weak)}. "
            "A weak fit is a poor basis for reading a change."
        )
    if floors:
        lines.append(
            "A peak is at the narrowest width the fit was allowed to use: "
            f"{_sample(floors)}."
        )
    if dropped:
        lines.append(f"Left out of the fit: {_sample(dropped)}.")
    if not lines:
        lines.append(
            "No channel was flagged for a weak fit, a rejected channel, or a peak at the narrowest allowed width."
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
            lines.append("Posterior peaks: " + "; ".join(_peak_phrase(item, (low, high)) for item in peaked) + ".")
        if quiet:
            lines.append(f"No posterior peak centered in {phrase}: {_sample(quiet)}.")
        others = [
            item.name for item in _ordered(fit.channels)
            if item.usable and item.name not in _POSTERIOR and item.name != channel
            and alpha_peak(item, (low, high)) is not None
        ]
        if others:
            lines.append(f"Other channels with a peak in {phrase}: {_sample(others)}.")
        elsewhere = _elsewhere(fit, low, high)
        if elsewhere:
            lines.append(f"Other rhythms in the {fit.fmin:g}–{fit.fmax:.0f} Hz fit: {elsewhere}.")
        lines.append(_exponent_sentence(fit, label))
        lines.append(
            "Peak height is the oscillatory peak above the aperiodic background. "
            "The exponent is that background slope. A change in band power can come from either one."
        )
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
        f"Compare is {spoken_left} against {spoken_right}. "
        f"Peak difference is {spoken_left} fitted peak height minus {spoken_right} fitted peak height, "
        "above the aperiodic background. "
        + _band_change(left, right),
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
        lines.append(" ".join(posterior))
    both = sum(1 for row in rows if row.alpha_power_ec is not None and row.alpha_power_eo is not None)
    only_left = sum(1 for row in rows if row.alpha_power_ec is not None and row.alpha_power_eo is None)
    only_right = sum(1 for row in rows if row.alpha_power_ec is None and row.alpha_power_eo is not None)
    neither = len(rows) - both - only_left - only_right
    lines.append(
        f"In-band peak on both sides: {both}. {_title_name(left)} only: {only_left}. "
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
        shape = _shape(fit.period_ms, fit.rise_decay, fit.peak_trough, fit.volt_amp)
        if shape:
            lines.append(f"All cycles: {shape}.")
        if fit.n_burst_cycles:
            burst_shape = _shape(fit.burst_period_ms, fit.burst_rise_decay, fit.burst_peak_trough, fit.burst_volt_amp)
            if burst_shape:
                lines.append(f"Burst cycles: {burst_shape}.")
        if fit.n_burst_episodes >= 2 and fit.burst_fraction is not None and fit.burst_fraction < 0.5:
            lines.append("The bursts are separate runs, and they cover less than half of the accepted time.")
        elif fit.burst_fraction is not None and fit.burst_fraction >= 0.8 and fit.n_burst_cycles:
            lines.append("Burst cycles cover most of the accepted time.")
        if fit.n_cycles and fit.n_burst_cycles == 0 and fit.burst_note:
            lines.append(fit.burst_note)
        if fit.notes:
            lines.append(fit.notes)
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
    usable = [item for item in fit.channels if item.usable and item.exponent is not None]
    if not usable:
        return f"{label} has no exponent to report."
    ordered = sorted(usable, key=lambda item: float(item.exponent))
    values = [float(item.exponent) for item in ordered]
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    flat = ordered[0]
    steep = ordered[-1]
    return (
        f"Exponent over {fit.fmin:g}–{fit.fmax:.0f} Hz: median {median:.2f} "
        f"across {len(usable)} channels. Flattest {flat.name} {flat.exponent:.2f}, "
        f"steepest {steep.name} {steep.exponent:.2f}."
    )


def _with_exponent(sentence: str, item: ChannelFit) -> str:
    if item.exponent is None:
        return sentence
    return f"{sentence} Exponent {item.exponent:.2f}."


def _elsewhere(fit: SpectrumFit, low: float, high: float) -> str:
    grouped: dict[str, list[str]] = {}
    for item in _ordered(fit.channels):
        if not item.usable:
            continue
        for center, height, _width in item.peaks:
            if low <= center <= high:
                continue
            name = _named_band(center)
            if name is None:
                continue
            grouped.setdefault(name, []).append(f"{item.name} {center:.2f} Hz height {height:.2f}")
    parts = []
    for name, _edge_low, _edge_high in NAMED_BANDS:
        found = grouped.get(name)
        if found:
            parts.append(f"{name} {_sample(found, limit=4)}")
    return "; ".join(parts)


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
    if percent is None:
        return ""
    if abs(percent) < 1.0:
        return "The Welch band change between the two recordings is about zero."
    return (
        f"The Welch band change is {abs(percent):.0f}% higher with {_higher_side(percent, left, right)} "
        "than with the other recording."
    )


def _reference_sentence(settings: FitSettings) -> str:
    if settings.reference == "average":
        return "Reference: average of the fitted channels."
    if settings.reference == "linked":
        return (
            "Reference: linked ears (A1 and A2, or M1 and M2) when the file has them. "
            "Otherwise the stored reference is kept."
        )
    return "Reference: as stored in the file."


def _slope_sentence(
    chosen: ChannelComparison,
    slopes: list[ChannelComparison],
    left: str,
    right: str,
) -> str:
    delta = float(chosen.exponent_delta)
    ordered = sorted(slopes, key=lambda row: abs(float(row.exponent_delta)), reverse=True)
    median = sorted(float(row.exponent_delta) for row in slopes)[len(slopes) // 2]
    largest = ", ".join(
        f"{row.name} {row.exponent_delta:+.2f}" for row in ordered[:3]
    )
    return (
        f"Exponent, {_mention(right)} minus {_mention(left)}, is {delta:+.2f} on {chosen.name}. "
        f"The median change is {median:+.2f}. Largest changes: {largest}. "
        "A change in the exponent is a broadband shift across the fit range."
    )


def _shape(
    period: float | None,
    rise: float | None,
    trough: float | None,
    voltage: float | None,
) -> str:
    parts = []
    if period is not None:
        parts.append(f"period {period:.0f} ms")
    if rise is not None:
        parts.append(f"rise–decay {rise:.2f}")
    if trough is not None:
        parts.append(f"peak–trough {trough:.2f}")
    if voltage is not None:
        parts.append(f"voltage {voltage:.1f} µV")
    return ", ".join(parts)


def _peak_phrase(item: ChannelFit, band: tuple[float, float]) -> str:
    peak = alpha_peak(item, band)
    if peak is None:
        return item.name
    return f"{item.name} {peak[0]:.2f} Hz height {peak[1]:.2f}"


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


def _sample(items: list[str], limit: int = 8) -> str:
    if len(items) <= limit:
        return _join(items)
    return f"{_join(items[:limit])}, and {len(items) - limit} more"


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
