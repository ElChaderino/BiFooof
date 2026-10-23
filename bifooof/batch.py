"""Headless load, fit, and export used by the command line."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from .analysis.compare import compare_spectra
from .analysis.cycles import BANDS, fit_cycles
from .analysis.settings import FitSettings
from .analysis.specparam_fit import fit_spectra
from .export import (
    burst_episode_rows, comparison_rows, cycle_detail_rows, cycle_summary_row, fit_metadata, peak_rows,
    save_figure, spectrum_rows, write_csv, write_json,
)
from .io.condition import CONDITION_LABELS, guess_condition
from .io.load import load_recording
from .io.recording import Recording
from .plots import render_comparison, render_cycles, render_spectrum

_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
_PREFERRED = ("O1", "O2", "Pz", "Cz", "Fp1")


def describe(recording: Recording) -> dict:
    return {
        "file": recording.source_path,
        "format": recording.source_format,
        "channels": recording.n_channels,
        "names": list(recording.ch_names),
        "sfreq_hz": recording.sfreq,
        "duration_s": recording.duration_s,
        "samples": recording.n_samples,
        "notes": list(recording.notes),
    }


def format_description(info: dict) -> str:
    names = ", ".join(info["names"])
    notes = " ".join(info["notes"])
    lines = [
        Path(info["file"]).name,
        f"  format {info['format']}   {info['channels']} ch   "
        f"{info['sfreq_hz']:.3g} Hz   {info['duration_s']:.1f} s   {info['samples']} samples",
        f"  channels {names}",
    ]
    if notes:
        lines.append(f"  {notes}")
    return "\n".join(lines)


def resolve_band(text: str) -> tuple[float, float]:
    key = text.strip().lower().replace("–", "-")
    if key in _BANDS:
        return _BANDS[key]
    for label, limits in BANDS:
        if key in label.lower():
            return limits
    parts = key.replace(",", "-").split("-")
    if len(parts) == 2:
        low, high = float(parts[0]), float(parts[1])
        if high > low > 0:
            return low, high
    known = ", ".join(_BANDS)
    raise ValueError(f"Unknown band {text!r}. Use one of {known}, or a range like 8-12.")


def pick_channel(recording: Recording, name: str | None) -> int:
    if name:
        for index, channel in enumerate(recording.ch_names):
            if channel.lower() == name.lower():
                return index
        known = ", ".join(recording.ch_names)
        raise ValueError(f"Channel {name!r} is not in this recording. Channels: {known}")
    for preferred in _PREFERRED:
        if preferred in recording.ch_names:
            return recording.ch_names.index(preferred)
    return 0


def run_file(path: str | Path, out_dir: str | Path, condition: str | None = None,
             channel: str | None = None, band: str = "alpha",
             cycles: bool = True, plots: bool = True,
             settings: FitSettings | None = None) -> list[Path]:
    """Fit one recording and write its tables and plots."""
    recording = load_recording(path)
    chosen = condition or guess_condition(recording.source_path) or "ec"
    if chosen not in CONDITION_LABELS:
        raise ValueError("Condition must be ec or eo.")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    index = pick_channel(recording, channel)
    limits = resolve_band(band)
    written: list[Path] = []

    fit_settings = settings or FitSettings()
    fit = fit_spectra(recording, fit_settings)
    written.append(write_csv(
        destination / "spectrum.csv",
        spectrum_rows(fit, chosen, recording.source_path),
    ))
    peaks = peak_rows(fit, chosen, recording.source_path)
    if peaks:
        written.append(write_csv(destination / "peaks.csv", peaks))
    written.append(write_json(
        destination / "metadata.json",
        fit_metadata(recording, fit_settings, fit, chosen),
    ))
    if plots:
        fig = Figure(figsize=(10, 7), dpi=120, facecolor="white")
        render_spectrum(fig, fit, index, CONDITION_LABELS[chosen])
        written.append(save_figure(fig, destination / "spectrum.png"))

    if cycles:
        cycle_fit = fit_cycles(recording, index, limits, fit_settings)
        written.append(write_csv(
            destination / "cycle_summary.csv",
            [cycle_summary_row(cycle_fit, chosen, recording.source_path)],
        ))
        details = cycle_detail_rows(cycle_fit, chosen)
        if details:
            written.append(write_csv(destination / "cycles.csv", details))
        episodes = burst_episode_rows(cycle_fit, chosen, recording.source_path)
        if episodes:
            written.append(write_csv(destination / "burst_episodes.csv", episodes))
        if plots:
            fig = Figure(figsize=(10, 3.4), dpi=120, facecolor="white")
            render_cycles(fig, cycle_fit)
            written.append(save_figure(fig, destination / "cycles.png"))
    return written


def compare_files(eyes_closed: str | Path, eyes_open: str | Path, out_dir: str | Path,
                  channel: str | None = None, band: str = "alpha",
                  cycles: bool = True, plots: bool = True,
                  settings: FitSettings | None = None) -> list[Path]:
    """Fit both conditions and write the comparison."""
    closed = load_recording(eyes_closed)
    opened = load_recording(eyes_open)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    name = _shared_channel(closed, opened, channel)
    limits = resolve_band(band)
    fit_settings = settings or FitSettings()
    closed_fit = fit_spectra(closed, fit_settings)
    opened_fit = fit_spectra(opened, fit_settings)
    rows = compare_spectra(
        closed_fit, opened_fit, fit_settings.alpha_band(), fit_settings.peak_width_low,
    )
    written = [
        write_csv(destination / "ec_spectrum.csv", spectrum_rows(closed_fit, "ec", closed.source_path)),
        write_csv(destination / "eo_spectrum.csv", spectrum_rows(opened_fit, "eo", opened.source_path)),
        write_csv(destination / "comparison.csv", comparison_rows(rows, fit_settings.alpha_band())),
        write_json(destination / "metadata.json", {
            "ec": fit_metadata(closed, fit_settings, closed_fit, "ec"),
            "eo": fit_metadata(opened, fit_settings, opened_fit, "eo"),
        }),
    ]
    for key, fit, recording in (("ec", closed_fit, closed), ("eo", opened_fit, opened)):
        peaks = peak_rows(fit, key, recording.source_path)
        if peaks:
            written.append(write_csv(destination / f"{key}_peaks.csv", peaks))
    if plots:
        fig = Figure(figsize=(10, 7), dpi=120, facecolor="white")
        render_comparison(
            fig, closed_fit, opened_fit, name, fit_settings.alpha_band(), fit_settings.peak_width_low,
        )
        written.append(save_figure(fig, destination / "comparison.png"))
        closed_index = closed.ch_names.index(name) if name in closed.ch_names else 0
        opened_index = opened.ch_names.index(name) if name in opened.ch_names else 0
        for key, fit, index in (("ec", closed_fit, closed_index), ("eo", opened_fit, opened_index)):
            fig = Figure(figsize=(10, 7), dpi=120, facecolor="white")
            render_spectrum(fig, fit, index, CONDITION_LABELS[key])
            written.append(save_figure(fig, destination / f"{key}_spectrum.png"))
    if cycles:
        summaries = []
        for key, recording in (("ec", closed), ("eo", opened)):
            if name not in recording.ch_names:
                continue
            cycle_fit = fit_cycles(recording, recording.ch_names.index(name), limits, fit_settings)
            summaries.append(cycle_summary_row(cycle_fit, key, recording.source_path))
            details = cycle_detail_rows(cycle_fit, key)
            if details:
                written.append(write_csv(destination / f"{key}_cycles.csv", details))
            episodes = burst_episode_rows(cycle_fit, key, recording.source_path)
            if episodes:
                written.append(write_csv(destination / f"{key}_burst_episodes.csv", episodes))
        if summaries:
            written.append(write_csv(destination / "cycle_comparison.csv", summaries))
    return written


def _shared_channel(closed: Recording, opened: Recording, name: str | None) -> str:
    if name:
        for recording in (closed, opened):
            if not any(channel.lower() == name.lower() for channel in recording.ch_names):
                known = ", ".join(recording.ch_names)
                raise ValueError(f"Channel {name!r} is missing from {Path(recording.source_path).name}. Channels: {known}")
        for channel in closed.ch_names:
            if channel.lower() == name.lower():
                return channel
    for preferred in _PREFERRED:
        if preferred in closed.ch_names and preferred in opened.ch_names:
            return preferred
    for channel in closed.ch_names:
        if channel in opened.ch_names:
            return channel
    return closed.ch_names[0]
