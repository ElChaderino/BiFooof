"""Inspectable quality, regional, stability, and alpha-candidate summaries."""
from dataclasses import replace
import numpy as np

from .compare import alpha_peak, band_power
from .prepare import R2_WEAK
from .settings import matched_settings
from .specparam_fit import fit_spectra
from ..jobs import checkpoint


def quality_rows(recordings, fits, reviewed=()):
    result = []
    for condition, source in recordings.items():
        fit = fits.get(condition)
        channels = {} if fit is None else {item.name: item for item in fit.channels}
        for name in source.ch_names:
            item = channels.get(name)
            support = {} if fit is None else fit.provenance.get("channels", {}).get(name, {})
            duration = source.duration_s if fit is None else fit.t_end - fit.t_start
            accepted = support.get("accepted_duration_s")
            segments = None
            if fit is not None and fit.welch_nperseg:
                width = fit.welch_nperseg
                segments = sum(max(0, 1 + (round((stop - start) * source.sfreq) - width) // max(1, width // 2))
                    for start, stop in support.get("accepted_spans_source_s", []))
            flags = []
            if item is None:
                flags.append("Fit pending or unavailable")
            elif not item.usable:
                flags.append(item.reason or "No fit")
            else:
                if item.r_squared is not None and item.r_squared < R2_WEAK:
                    flags.append("Low R²; inspect residual")
                if item.peak_at_width_floor:
                    flags.append("Peak at width limit")
                if not item.peaks:
                    flags.append("No fitted peaks")
            result.append({"condition": condition, "channel": name,
                "accepted_s": accepted, "excluded_percent": None if accepted is None or duration <= 0 else 100 * max(0, 1 - accepted / duration),
                "welch_segments": segments, "epochs": None if item is None else item.n_epochs,
                "r_squared": None if item is None else item.r_squared, "fit_error": None if item is None else item.error,
                "reference_requested": None if fit is None or fit.settings is None else fit.settings.reference,
                "reference_applied": None if fit is None else fit.provenance.get("reference_applied", "unknown"),
                "review": "Reviewed" if f"{condition}:{name}" in reviewed else "Unreviewed",
                "flags": "; ".join(flags)})
    return result


def comparison_checks(recordings, fits):
    if not all(key in fits for key in ("ec", "eo")):
        return [{"level": "info", "check": "Comparison", "detail": "Both completed spectra are needed."}]
    left, right = fits["ec"], fits["eo"]
    checks = []
    def add(level, check, detail):
        checks.append({"level": level, "check": check, "detail": detail})
    common = set(item.name for item in left.channels if item.usable) & set(item.name for item in right.channels if item.usable)
    add("info" if common else "incompatible", "Shared fitted channels", ", ".join(sorted(common)) or "None")
    for attribute in ("fmin", "fmax", "aperiodic_mode", "segment_s", "epoch_s"):
        a, b = getattr(left, attribute), getattr(right, attribute)
        if a != b:
            add("caution", attribute, f"EC {a}; EO {b}")
    a, b = left.provenance, right.provenance
    for field in ("reference_applied", "reference_contributors"):
        if a.get(field) != b.get(field):
            add("caution", field.replace("_", " "), f"EC {a.get(field)}; EO {b.get(field)}")
    if left.settings and right.settings:
        for name in ("peak_width_low", "peak_width_high", "max_n_peaks", "min_peak_height", "peak_threshold", "amplitude_limit_uv"):
            if getattr(left.settings, name) != getattr(right.settings, name):
                add("caution", name, f"EC {getattr(left.settings, name)}; EO {getattr(right.settings, name)}")
        for key, fit in fits.items():
            applied = fit.provenance.get("reference_applied")
            if fit.settings and applied != fit.settings.reference:
                add("caution", "Reference fallback", f"{key}: requested {fit.settings.reference}; applied {applied}")
    for name in sorted(common):
        ca, cb = a.get("channels", {}).get(name, {}), b.get("channels", {}).get(name, {})
        da, db = ca.get("accepted_duration_s"), cb.get("accepted_duration_s")
        if da is not None and db is not None and abs(da - db) > .01:
            add("caution", f"Usable signal: {name}", f"EC {da:.2f} s; EO {db:.2f} s. Select suitable windows explicitly.")
    if "ec" in recordings and "eo" in recordings:
        units = [recordings[key].import_info.get("source_units", "Unknown") for key in ("ec", "eo")]
        if units[0] != units[1] or any("unknown" in str(value).lower() or value == {} for value in units):
            add("caution", "Source units", f"EC {units[0]}; EO {units[1]}. Analysis values use µV after import conversion.")
        if recordings["ec"].sfreq != recordings["eo"].sfreq:
            add("info", "Sampling rates", "Rates differ; inspect common frequency coverage and resolution for this measure.")
        if any(not source.import_info.get("stored_reference") or source.import_info.get("stored_reference") == "Unknown" for source in recordings.values()):
            add("caution", "Stored reference", "Original reference information is unknown on at least one recording.")
    add("info", "Fit windows", f"EC {left.t_start:g}–{left.t_end:g} s; EO {right.t_start:g}–{right.t_end:g} s")
    return checks


def epoch_rows(fit, condition=""):
    rows = []
    for epoch in fit.epochs or [fit]:
        for item in epoch.channels:
            peak = alpha_peak(item, fit.settings.alpha_band() if fit.settings else (8, 12))
            rows.append({"condition": condition, "channel": item.name, "start_s": epoch.t_start,
                "end_s": epoch.t_end, "usable": item.usable, "reason": item.reason,
                "exponent": item.exponent, "peak_hz": None if peak is None else peak[0],
                "peak_height": None if peak is None else peak[1], "r_squared": item.r_squared,
                "accepted_s": epoch.provenance.get("channels", {}).get(item.name, {}).get("accepted_duration_s")})
    return rows


def stability_summary(rows):
    summaries = []
    for name in sorted({row["channel"] for row in rows}):
        chosen = [row for row in rows if row["channel"] == name]
        for metric in ("exponent", "peak_hz", "peak_height", "burst_fraction", "burst_episodes"):
            values = [(row["start_s"], row.get(metric)) for row in chosen if row.get(metric) is not None and np.isfinite(row[metric])]
            if not values:
                continue
            values.sort()
            array = np.array([value for _time, value in values])
            middle = (min(row["start_s"] for row in chosen) + max(row["end_s"] for row in chosen)) / 2
            first = [value for moment, value in values if moment < middle]
            second = [value for moment, value in values if moment >= middle]
            summaries.append({"channel": name, "metric": metric, "n": len(array),
                "total_windows": len(chosen), "peak_windows": sum(row.get("peak_hz") is not None for row in chosen),
                "median": float(np.median(array)), "q1": float(np.percentile(array, 25)), "q3": float(np.percentile(array, 75)),
                "first_half_median": None if not first else float(np.median(first)),
                "second_half_median": None if not second else float(np.median(second)),
                "definition": "Nonoverlapping window variability; not a confidence interval"})
    return summaries


def run_stability(recording, settings, seconds=30.0, condition=""):
    from .cycles import fit_cycles
    fit = fit_spectra(recording, replace(settings, epoch_s=seconds))
    rows = epoch_rows(fit, condition)
    epochs = fit.epochs or [fit]
    for position, epoch in enumerate(epochs):
        for index, name in enumerate(recording.ch_names):
            checkpoint(position, len(epochs), f"Window cycles: {name}, {position + 1}/{len(epochs)}")
            try:
                cycles = fit_cycles(recording, index, settings.alpha_band(), replace(settings, epoch_s=None, t_start=epoch.t_start, t_length=epoch.t_end - epoch.t_start))
                row = next(row for row in rows if row["channel"] == name and row["start_s"] == epoch.t_start)
                row.update(burst_fraction=cycles.burst_fraction, burst_episodes=cycles.n_burst_episodes, cycle_failures=len(cycles.failures))
            except ValueError:
                continue
    return rows


def alpha_candidates(fit, channels, search=(7.0, 14.0)):
    candidates = []
    for item in fit.channels:
        if item.name not in channels or not item.usable or item.r_squared is None or item.r_squared < R2_WEAK:
            continue
        peak = alpha_peak(item, search)
        if peak:
            candidates.append({"channel": item.name, "peak_hz": peak[0], "height_log10": peak[1], "bandwidth_hz": peak[2],
                "r_squared": item.r_squared, "epochs": item.n_epochs, "width_limit": item.peak_at_width_floor,
                "resolution_hz": None if item.freqs is None or len(item.freqs) < 2 else float(np.median(np.diff(item.freqs)))})
    return candidates


def regional_rows(fits, regions, band):
    result = []
    common = None
    if len(fits) > 1:
        common = set.intersection(*[{item.name for item in fit.channels if item.usable} for fit in fits.values()])
    for condition, fit in fits.items():
        for region, members in regions.items():
            items = [item for item in fit.channels if item.usable and item.name in members and (common is None or item.name in common)]
            def median(values):
                finite = [value for value in values if value is not None and np.isfinite(value)]
                return None if not finite else float(np.median(finite))
            powers = [band_power(item, band) for item in items]
            peaks = [alpha_peak(item, band) for item in items]
            finite_power = [power for power in powers if power is not None and np.isfinite(power)]
            result.append({"condition": condition, "region": region, "channels": ", ".join(item.name for item in items),
                "missing": ", ".join(name for name in members if name not in {item.name for item in items}),
                "exponent_median": median(item.exponent for item in items),
                "peak_hz_median": median(peak[0] for peak in peaks if peak),
                "peak_height_median": median(peak[1] for peak in peaks if peak),
                "mean_band_psd_uv2_hz": None if not finite_power else float(np.mean(finite_power)),
                "definition": "Summary of channel parameters; mean linear band PSD; common fitted channels across conditions"})
    return result


def asymmetry(rows, left, right):
    result = []
    for condition in sorted({row["condition"] for row in rows}):
        a = next((row for row in rows if row["condition"] == condition and row["region"] == left), None)
        b = next((row for row in rows if row["condition"] == condition and row["region"] == right), None)
        pa, pb = (None if row is None else row["mean_band_psd_uv2_hz"] for row in (a, b))
        value = None if pa is None or pb is None or pa <= 0 or pb <= 0 else float(np.log10(pa) - np.log10(pb))
        result.append({"condition": condition, "left": left, "right": right, "log10_power_ratio": value,
            "definition": "log10(mean channel band PSD in left region) - log10(mean channel band PSD in right region)"})
    return result
