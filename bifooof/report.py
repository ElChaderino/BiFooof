"""Offline reviewed reports, printed with Qt's existing PDF engine."""
import base64
from html import escape
from pathlib import Path

from . import __version__


MEASURES = {
    "exponent": {"units": "unitless", "definition": "Aperiodic exponent over the declared fit range"},
    "peak_hz": {"units": "Hz", "definition": "Fitted in-band peak center; missing when no peak was found"},
    "peak_height": {"units": "log10 power", "definition": "Peak height above the aperiodic background"},
    "mean_band_psd": {"units": "follows the reference", "definition": "Mean linear Welch PSD over the declared band. µV²/Hz after a voltage reference, scaled µV²/Hz after the global scale factor, and µV²/(m⁴·Hz) after current source density."},
    "band_change_percent": {"units": "%", "definition": "100 * (left - right) / right; missing for absent or nonpositive denominator"},
    "burst_cycles": {"units": "cycles", "definition": "Cycles labeled as part of a burst"},
    "burst_episodes": {"units": "episodes", "definition": "Separate contiguous runs of burst cycles"},
    "burst_fraction": {"units": "fraction", "definition": "Burst duration divided by accepted signal duration"},
    "accepted_s": {"units": "s", "definition": "Accepted signal duration contributing to the declared analysis"},
    "excluded_percent": {"units": "%", "definition": "Percentage of the selected window excluded from that channel's analysis"},
    "r_squared": {"units": "unitless", "definition": "Spectral model goodness of fit; inspect residuals and signal support alongside it"},
    "fit_error": {"units": "log10 power", "definition": "Mean absolute spectral model error"},
    "period_ms": {"units": "ms", "definition": "Median cycle period; burst-prefixed fields restrict this to burst cycles"},
    "volt_amp": {"units": "follows the reference", "definition": "Median cycle peak-to-trough amplitude. µV after a voltage reference, scaled µV after the global scale factor, and µV/m² after current source density."},
    "rise_decay": {"units": "fraction", "definition": "Median fraction of each cycle spent rising"},
    "peak_trough": {"units": "fraction", "definition": "Median fraction of each cycle spent in its peak phase"},
    "log10_power_ratio": {"units": "log10 ratio", "definition": "log10(mean left-region band PSD / mean right-region band PSD), using declared common channel sets"},
    "baseline_delta": {"units": "same as selected measure", "definition": "Saved session value minus the selected baseline project's value for the same channel and condition"},
}


def html_table(rows):
    if not rows:
        return "<p>No completed results for this section.</p>"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return "—" if value != value or value in {float("inf"), float("-inf")} else f"{value:.6g}"
        return str(value)
    return "<table><thead><tr>" + "".join(f"<th>{escape(key.replace('_', ' '))}</th>" for key in fields) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(cell(row.get(key)))}</td>" for key in fields) + "</tr>" for row in rows) + "</tbody></table>"


def report_html(snapshot, options=None):
    options = {"notes": True, "quality": True, "results": True, "plots": True, "paths": False, **(options or {})}
    title = snapshot.get("alias") or "EEG review"
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>" + escape(title) + "</title><style>"
        "body{font:10pt 'Segoe UI',Arial,sans-serif;color:#16232f;margin:24px}h1{font-size:20pt;color:#1c4e78}"
        "h2{font-size:14pt;color:#1c4e78;margin-top:22px}table{border-collapse:collapse;width:100%;font-size:9pt}"
        "td,th{border:1px solid #d2dbe6;padding:5px;text-align:left}th{background:#edf1f6}"
        "p{line-height:1.4}img{max-width:100%;height:auto}thead{display:table-header-group}"
        "</style></head><body>", f"<h1>{escape(title)}</h1>",
        f"<p>{escape(snapshot.get('date', ''))} · BiFooof {__version__} · reviewed analysis snapshot</p>"]
    if snapshot.get("condition_labels"):
        parts.append("<p>Condition labels: " + escape("; ".join(f"{key.upper()} = {value}" for key, value in snapshot["condition_labels"].items())) + ".</p>")
    if options["paths"]:
        parts.append("<h2>Sources</h2>" + html_table(snapshot.get("sources", [])))
    if options["notes"]:
        parts.append("<h2>Reviewer observations</h2><p>" + escape(snapshot.get("notes", "")).replace("\n", "<br>") + "</p>")
    if options["quality"]:
        parts.append("<h2>Signal support and comparison checks</h2>" + html_table(snapshot.get("checks", [])))
        quality = [{key: row.get(key) for key in ("condition", "channel", "accepted_s", "excluded_percent", "reference_applied", "r_squared", "flags", "review")} for row in snapshot.get("quality", [])]
        parts.append(html_table(quality))
    if options["results"]:
        findings = snapshot.get("findings", "")
        if not options["paths"]:
            for row in snapshot.get("sources", []):
                path = row.get("path", "")
                if path:
                    findings = findings.replace(path, row["condition"]).replace(Path(path).name, row["condition"])
        parts.append("<h2>Calculated findings</h2><p>" + escape(findings).replace("\n", "<br>") + "</p>")
        parts.append("<h2>Applied processing settings</h2>" + html_table([{"setting": key, "value": value} for key, value in snapshot.get("settings", {}).items()]))
        parts.append("<h2>Software versions</h2>" + html_table([snapshot.get("versions", {})]))
        parts.append("<h2 style='page-break-before:always'>Measurement definitions</h2>" + html_table([{"measure": key, **value} for key, value in MEASURES.items()]))
    if options["plots"]:
        for picture in snapshot.get("plots", []):
            if "plot_titles" in options and picture["title"] not in options["plot_titles"]:
                continue
            width = 650
            height = round(width * picture["height"] / max(1, picture["width"]))
            parts.append(f"<div style='page-break-before:always'><h2>{escape(picture['title'])}</h2><img width='{width}' height='{height}' src='data:image/png;base64,{picture['data']}'></div>")
    parts.append("<p>Missing fitted peaks remain missing. Reviewer observations and calculated measurements are separate sections.</p></body></html>")
    return "".join(parts)


def write_report(folder, snapshot, options=None, pdf=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text = report_html(snapshot, options)
    html = folder / "report.html"
    html.write_text(text, encoding="utf-8")
    paths = [html]
    if pdf:
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QFont, QPageSize, QPdfWriter, QTextDocument
        destination = folder / "report.pdf"
        writer = QPdfWriter(str(destination))
        writer.setTitle(snapshot.get("alias") or "EEG review")
        writer.setCreator(f"BiFooof {__version__}")
        writer.setResolution(96)
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setPageMargins(QMarginsF(14, 14, 14, 14))
        document = QTextDocument()
        document.setDefaultFont(QFont("Segoe UI", 10))
        document.setHtml(text)
        document.setPageSize(QSizeF(writer.width(), writer.height()))
        document.print_(writer)
        del writer
        if not destination.exists() or destination.stat().st_size < 100:
            raise OSError("The PDF writer did not produce a complete document.")
        paths.append(destination)
    return paths
