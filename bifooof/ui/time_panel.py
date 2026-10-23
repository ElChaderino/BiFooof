"""Time-resolved views for the recording already open on the other pages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..analysis.compare import alpha_peak, compare_pair
from ..analysis.coherence import coherence_pair
from ..analysis.cycles import CycleFit, fit_cycles
from ..analysis.pac import phase_amplitude
from ..analysis.quality import QualitySpan, mask_spans, quality_spans
from ..analysis.settings import FitSettings, band_phrase, matched_settings
from ..analysis.shape import burst_stats
from ..analysis.timefreq import (
    SlidingFit, Spectrogram, align_to_events, block_band_power, channel_series,
    channel_spectrogram, exponent_spread, sliding_spectra, track_peaks, track_summary, window_at,
)
from ..io.condition import condition_in_text
from ..io.recording import Recording
from .canvas import MplCanvas
from .time_draw import (
    draw_coherence, draw_durations, draw_pac, draw_quality, draw_raster,
    draw_reactivity, draw_scatter, draw_shape, draw_spectrogram, draw_trajectory,
    draw_trajectory_with_maps, duration_caption,
)
from .workers import CallWorker, launch, wait_for

_TIME_MODES = (
    ("Spectrogram", "spectrogram"),
    ("Exponent and peaks", "trajectory"),
    ("Bursts", "raster"),
    ("Burst lengths", "durations"),
    ("Cycle shape", "shape"),
    ("Amplitude and frequency", "scatter"),
    ("Reactivity", "reactivity"),
    ("Coherence", "coherence"),
    ("Coupling", "coupling"),
    ("Quality", "quality"),
)


class TimePanel(QWidget):
    """One tab, several views, all on the fit-bar settings and the selected channel."""

    inspect_at = Signal(float, float)
    window_requested = Signal(float, float)
    exclusions_changed = Signal(object)
    channel_requested = Signal(str)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.recording: Recording | None = None
        self.recordings: dict[str, Recording] = {}
        self.sources: dict[str, Recording] = {}
        self.continuous: Recording | None = None
        self.condition = "ec"
        self.settings = FitSettings()
        self.cycle_fit: CycleFit | None = None
        self._selected = 0
        self._generation = 0
        self._worker: CallWorker | None = None
        self._cache: dict = {}
        self._source_id: dict[str, int] = {}
        self._mode = "spectrogram"
        self._result = None
        self._sliding: SlidingFit | None = None
        self._sliding_id: tuple | None = None
        self._inspection: tuple[float, float] | None = None
        self.excluded: dict[str, list[QualitySpan]] = {}
        self.pair_cycles: dict[str, CycleFit] = {}
        self.spectra: dict = {}
        self.pair_names = ("Eyes closed", "Eyes open", "EC", "EO")
        self._partner: SlidingFit | None = None
        self._slidings: dict[str, SlidingFit] = {}
        self._span = None
        self._press: tuple[float, float | None] | None = None
        self._posterior_for: int | None = None
        self._quality_rows: list[QualitySpan] = []

        self.mode = QComboBox()
        for label, key in _TIME_MODES:
            self.mode.addItem(label, key)
        self.mode.setToolTip("Which time view to show. All of them use the fit bar.")
        self.mode.currentIndexChanged.connect(self._kick)

        self.spec_window = _choice((("1 s", 1.0), ("2 s", 2.0), ("4 s", 4.0)), 1)
        self.spec_window.setToolTip("Spectrogram window. 2 s is 0.5 Hz. Shorter windows locate a burst less sharply than the step size suggests.")
        self.spec_overlap = _choice((("50%", 0.5), ("75%", 0.75)), 1)
        self.spec_overlap.setToolTip("Overlap of successive spectrogram windows.")
        self.slide_window = _choice((("10 s", 10.0), ("30 s", 30.0), ("60 s", 60.0)), 1)
        self.slide_window.setToolTip("Length of each specparam window. 10 s with a 4 s Welch segment has few averages.")
        self.slide_overlap = _choice((("None", 0.0), ("50%", 0.5)), 1)
        self.slide_overlap.setToolTip("Overlap of successive specparam windows.")
        self.lock_color = QCheckBox("Lock color")
        self.lock_color.setToolTip(
            "Keep this color range. Unchecked, the range follows the 2nd and 98th percentile of the picture."
        )
        self.lock_color.toggled.connect(self._on_color_lock)
        self._color_limits: tuple[float, float] | None = None
        self.color_low = _decibel("Minimum color, in dB relative to 1 µV²/Hz.")
        self.color_high = _decibel("Maximum color, in dB relative to 1 µV²/Hz.")
        self.color_low.valueChanged.connect(self._on_color_range)
        self.color_high.valueChanged.connect(self._on_color_range)
        self.top_trace = _choice((("Exponent", "exponent"), ("Offset", "offset"), ("Knee", "knee")), 0)
        self.top_trace.setToolTip(
            "The top trace. Offset and knee come from the same windows. A fixed slope has no knee."
        )
        self.top_trace.currentIndexChanged.connect(self._redraw_only)
        self.map_which = _choice((("Median", "median"), ("Spread", "spread")), 0)
        self.map_which.setToolTip(
            "Which map to draw when every channel is on. The colors include the other open file when that fit exists."
        )
        self.map_which.currentIndexChanged.connect(self._redraw_only)
        self.picks = QListWidget()
        self.picks.setMaximumHeight(86)
        self.picks.setMaximumWidth(132)
        self.picks.setToolTip("Checked channels are the burst rows. Every channel ignores this list.")
        self.picks.itemChanged.connect(self._kick)
        self.react_pre = _seconds(10.0, "Seconds before each eye mark.")
        self.react_post = _seconds(20.0, "Seconds after each eye mark.")
        self.react_pre.valueChanged.connect(self._kick)
        self.react_post.valueChanged.connect(self._kick)
        self.react_channel = QComboBox()
        self.react_channel.setToolTip("Channel for this view. A posterior channel is chosen when the file has one.")
        self.react_channel.currentIndexChanged.connect(self._react_changed)
        self.amp_channel = QComboBox()
        self.amp_channel.setToolTip("Amplitude channel. Phase stays on the channel selected in the rest of the window.")
        self.amp_channel.currentIndexChanged.connect(self._kick)
        self.bursts = QCheckBox("Bursts")
        self.bursts.setChecked(True)
        self.bursts.setToolTip("On the spectrogram, draw burst episodes. On the cycle views, hide cycles that are not in a burst.")
        self.every = QCheckBox("Every channel")
        self.every.setToolTip("Fit every channel. The exponent map needs this. It takes longer.")
        self.align = _choice((("Trough", "trough"), ("Peak", "peak")), 0)
        self.align.setToolTip("Line the cycles up on the trough or on the peak. The normalized view resamples trough to trough either way.")
        self.normalized = QCheckBox("Normalized")
        self.normalized.setToolTip("Resample each cycle onto one axis from trough to trough. Timing differences are removed.")
        self.other = QComboBox()
        self.other.setToolTip("Second channel for coherence. Both use the reference on the fit bar.")
        self.phase_low = _hertz(4.0)
        self.phase_high = _hertz(8.0)
        self.amp_low = _hertz(30.0)
        self.amp_high = _hertz(45.0)
        self.phase_low.setToolTip("Low edge of the phase band.")
        self.phase_high.setToolTip("High edge of the phase band.")
        self.amp_low.setToolTip("Low edge of the amplitude band.")
        self.amp_high.setToolTip("High edge of the amplitude band.")
        self.comod = QCheckBox("Map")
        self.comod.setToolTip("Coarse map of the modulation index. Surrogates are only for the bands above. The map is not corrected across cells.")
        self.stage = QPushButton("Set fit window")
        self.stage.setObjectName("Secondary")
        self.stage.setToolTip("Write the clicked stretch into the fit bar. Press Refit to run it.")
        self.stage.clicked.connect(self._stage)
        self.leave = QPushButton("Leave out")
        self.leave.setObjectName("Secondary")
        self.leave.setToolTip("Keep the selected stretch out of later fits. The file on disk is not changed.")
        self.leave.clicked.connect(self._leave_out)
        self.restore = QPushButton("Put back")
        self.restore.setObjectName("Secondary")
        self.restore.setToolTip("Put the selected stretch back into the fit.")
        self.restore.clicked.connect(self._restore)

        for widget in (
            self.spec_window, self.spec_overlap, self.slide_window, self.slide_overlap,
            self.bursts, self.every, self.align, self.normalized, self.other, self.comod,
        ):
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._kick)
            else:
                widget.currentIndexChanged.connect(self._kick)
        for box in (self.phase_low, self.phase_high, self.amp_low, self.amp_high):
            box.valueChanged.connect(self._kick)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        view = QLabel("View")
        view.setObjectName("Muted")
        controls.addWidget(view)
        controls.addWidget(self.mode)
        controls.addWidget(self.spec_window)
        controls.addWidget(self.spec_overlap)
        controls.addWidget(self.slide_window)
        controls.addWidget(self.slide_overlap)
        controls.addWidget(self.lock_color)
        controls.addWidget(self.color_low)
        controls.addWidget(self.color_high)
        controls.addWidget(self.top_trace)
        controls.addWidget(self.map_which)
        controls.addWidget(self.picks)
        controls.addWidget(self.react_pre)
        controls.addWidget(self.react_post)
        controls.addWidget(self.react_channel)
        controls.addWidget(self.bursts)
        controls.addWidget(self.every)
        controls.addWidget(self.align)
        controls.addWidget(self.normalized)
        controls.addWidget(self.other)
        controls.addWidget(self.amp_channel)
        controls.addWidget(self.phase_low)
        controls.addWidget(self.phase_high)
        controls.addWidget(self.amp_low)
        controls.addWidget(self.amp_high)
        controls.addWidget(self.comod)
        controls.addWidget(self.leave)
        controls.addWidget(self.restore)
        controls.addStretch(1)
        controls.addWidget(self.stage)

        self.caption = QLabel("Open a recording. This page uses the same fit window, reference, and rejected samples as the spectrum.")
        self.caption.setObjectName("Muted")
        self.caption.setWordWrap(True)
        self.canvas = MplCanvas(width=8.0, height=5.6)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Channel", "Start", "End", "Kind", "Level"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(168)
        self.table.setAlternatingRowColors(True)
        self.table.setVisible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(controls)
        root.addWidget(self.caption)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.table)
        self._apply_visibility()
        self._empty("Open a recording.")

    def set_recording(
        self,
        recording: Recording,
        condition: str = "ec",
        source: Recording | None = None,
        recordings: dict | None = None,
        continuous: Recording | None = None,
    ) -> None:
        self.recording = recording
        self.condition = condition
        if recordings is not None:
            self.recordings = recordings
        self.continuous = continuous
        if source is not None:
            if self._source_id.get(condition) != id(source):
                self.excluded.pop(condition, None)
                self._inspection = None
                self._cache.clear()
            self.sources[condition] = source
            self._source_id[condition] = id(source)
        self._fill_other()
        self._fill_picks()
        self._fill_amp()
        self._fill_react_channel()
        self._kick()

    def set_pair_cycles(self, fits: dict[str, CycleFit], names=None) -> None:
        self.pair_cycles = dict(fits)
        if names is not None:
            self.pair_names = tuple(names)
        if self._mode in {"durations", "scatter"}:
            self._draw()

    def set_cycle_fit(self, fit: CycleFit | None) -> None:
        self.cycle_fit = fit
        if self._mode == "spectrogram" and not isinstance(self._result, Spectrogram):
            return
        if self._mode in {"spectrogram", "shape", "durations", "scatter"} or (
            self._mode == "raster" and not self.every.isChecked()
        ):
            self._draw()

    def set_selected(self, index: int) -> None:
        if index == self._selected:
            return
        self._selected = index
        self._fill_other()
        self._fill_picks()
        self._fill_amp()
        self._fill_react_channel()
        self._kick()

    def apply_settings(self, settings: FitSettings) -> None:
        self.settings = settings
        self._cache.clear()
        self._sliding = None
        self._sliding_id = None
        self._slidings.clear()
        self._kick()

    def shutdown(self) -> None:
        wait_for(self)

    def rows(self) -> list[dict]:
        result = self._result
        channel = self._channel_name()
        if isinstance(result, Spectrogram):
            rows = []
            key = self._settings_key()
            rate = 0.0 if self.recording is None else float(self.recording.sfreq)
            for index, moment in enumerate(result.times_s):
                column = result.power_db[:, index]
                sample = int(round(float(moment) * rate))
                for freq, level in zip(result.freqs_hz, column):
                    if not np.isfinite(level):
                        continue
                    rows.append({
                        "settings_key": key,
                        "channel": result.channel,
                        "time_s": round(float(moment), 3),
                        "sample": sample,
                        "freq_hz": round(float(freq), 3),
                        "power_db": round(float(level), 3),
                    })
            return rows
        if isinstance(result, SlidingFit):
            series = channel_series(result, channel, self.settings.alpha_band())
            found = []
            key = self._settings_key()
            rate = 0.0 if self.recording is None else float(self.recording.sfreq)
            for index, moment in enumerate(series.times_s):
                found.append({
                    "settings_key": key,
                    "channel": channel,
                    "time_s": round(float(moment), 3),
                    "sample": int(round(float(moment) * rate)),
                    "exponent": _cell(series.exponent[index]),
                    "offset": _cell(series.offset[index]),
                    "knee": _cell(None if series.knee is None else series.knee[index]),
                    "peak_hz": _cell(series.peak_hz[index]),
                    "peak_height": _cell(series.peak_height[index]),
                    "bandwidth_hz": _cell(series.peak_bandwidth[index]),
                    "r_squared": _cell(series.r_squared[index]),
                    "artifact_fraction": _cell(series.artifact_fraction[index]),
                })
            return found
        if self._mode == "raster":
            exported = []
            for fit in self._raster_fits():
                for episode in fit.episodes:
                    freq = None
                    if episode.period_ms is not None and episode.period_ms > 0:
                        freq = round(1000.0 / episode.period_ms, 3)
                    exported.append({
                        "settings_key": self._settings_key(),
                        "channel": fit.channel,
                        "start_s": round(episode.start_s, 3),
                        "duration_s": round(episode.end_s - episode.start_s, 3),
                        "cycles": episode.n_cycles,
                        "amplitude_uv": episode.volt_amp,
                        "frequency_hz": freq,
                    })
            return exported
        if self._mode in {"shape", "durations", "scatter"} and isinstance(self.cycle_fit, CycleFit):
            fit = self.cycle_fit
            if self._mode == "durations":
                stats = burst_stats(fit)
                return [{
                    "settings_key": self._settings_key(),
                    "channel": fit.channel,
                    "episodes": stats.n_episodes,
                    "median_s": stats.median_s,
                    "q1_s": stats.q1_s,
                    "q3_s": stats.q3_s,
                    "per_minute": stats.per_minute,
                    "occupancy": stats.occupancy,
                    "median_cycles": stats.median_cycles,
                    "median_gap_s": None if stats.gaps_s.size == 0 else float(np.median(stats.gaps_s)),
                }]
            key = self._settings_key()
            return [
                {"settings_key": key, "channel": fit.channel, "cycle": index + 1, **row}
                for index, row in enumerate(fit.cycles)
            ]
        if self._mode == "quality":
            return [
                {
                    "channel": span.channel,
                    "start_s": span.start_s,
                    "end_s": span.end_s,
                    "kind": span.kind,
                    "level": span.level,
                    "note": span.note,
                }
                for span in self._quality_rows
            ]
        if isinstance(result, dict) and result.get("kind") == "single":
            times, power = result.get("single", (np.zeros(0), np.zeros(0)))
            return [
                {"channel": channel, "time_s": float(moment), "band_power": None if not np.isfinite(value) else float(value)}
                for moment, value in zip(times, power)
            ]
        return []

    def _kick(self, *_args) -> None:
        self._mode = str(self.mode.currentData() or "spectrogram")
        self._apply_visibility()
        if self.recording is None:
            self.caption.setText("Open a recording. This page uses the same fit window, reference, and rejected samples as the spectrum.")
            self._empty("Open a recording.")
            return
        if self._mode == "reactivity" and self._offer_posterior():
            return
        if self._local_mode():
            self._result = self.cycle_fit
            self._draw()
            return
        key = self._cache_key()
        cached = self._accept(self._cache.get(key))
        if cached is not None:
            self._result = cached
            if self._mode == "trajectory" and isinstance(cached, SlidingFit):
                self._sliding = cached
                self._sliding_id = key
                self._slidings[self.condition] = cached
            self._draw()
            return
        self._generation += 1
        generation = self._generation
        self.caption.setText(self._waiting())
        self.status.emit(self.caption.text())
        worker = CallWorker(
            generation, _job,
            (
                self._mode, self.recording, self._fit_settings(),
                self._react_index() if self._mode == "reactivity" else self._index(),
                self._options(),
            ),
        )
        worker.succeeded.connect(self._done)
        worker.failed.connect(self._failed)
        self._worker = worker
        launch(self, worker)

    def _done(self, generation: int, result: object) -> None:
        if generation != self._generation:
            return
        result = self._accept(result)
        self._cache[self._cache_key()] = self._stored(result)
        self._result = result
        if self._mode == "trajectory" and isinstance(result, SlidingFit):
            self._sliding = result
            self._sliding_id = self._cache_key()
            self._slidings[self.condition] = result
        self._draw()
        self.status.emit("Ready.")

    def _failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        detail = message.strip().splitlines()[-1] if message.strip() else "That view failed."
        self.caption.setText(detail)
        self._empty(detail)
        self.status.emit(detail)

    def _draw(self) -> None:
        if self.recording is None:
            return
        fig = self.canvas.fig
        mode = self._mode
        try:
            if mode == "spectrogram" and isinstance(self._result, Spectrogram):
                window = None
                sliding = self._sliding_current()
                if self._inspection is not None:
                    if sliding is not None:
                        window = self._clicked_window(sliding, self._channel_name())
                    else:
                        window = (None, None, "The Spectrum tab is still the fit-bar window.")
                draw_spectrogram(
                    fig, self._result, self._episodes(), self._inspection, self._raw_stretch(),
                    self._overlay_tracks(), self._color_limits, window,
                )
            elif mode == "trajectory" and isinstance(self._result, SlidingFit):
                self._draw_trajectory(fig, self._result)
            elif mode == "raster":
                fits = self._raster_fits()
                draw_raster(fig, fits, self._inspection)
                self._result = fits
                self._fill_episode_table(fits)
            elif mode == "durations" and isinstance(self.cycle_fit, CycleFit):
                draw_durations(fig, self._duration_groups())
            elif mode == "shape" and isinstance(self.cycle_fit, CycleFit):
                draw_shape(
                    fig, self.cycle_fit, str(self.align.currentData()),
                    self.bursts.isChecked(), self.normalized.isChecked(),
                )
            elif mode == "scatter" and isinstance(self.cycle_fit, CycleFit):
                draw_scatter(fig, self._scatter_groups(), self.bursts.isChecked())
            elif mode == "reactivity" and isinstance(self._result, dict):
                payload = self._result
                if payload.get("kind") == "separate":
                    payload = dict(payload)
                    payload["left_label"] = self.pair_names[0]
                    payload["right_label"] = self.pair_names[1]
                draw_reactivity(fig, payload, self._inspection)
            elif mode == "coherence":
                current, other, other_label = self._coherence_parts()
                if current is None:
                    self._empty(self._waiting())
                    return
                draw_coherence(fig, current, self._coherence_raw(), other, other_label)
            elif mode == "coupling" and self._result is not None:
                draw_pac(fig, self._result)
            elif mode == "quality":
                self._draw_quality(fig)
            elif self._local_mode():
                self._empty(self._cycle_wait())
                self.caption.setText(self._cycle_wait())
                return
            else:
                self._empty(self._waiting())
                return
        except Exception as exc:
            self._empty(str(exc))
            self.caption.setText(str(exc))
            return
        self._write_caption()
        self.canvas.draw_idle()

    def _draw_trajectory(self, fig, sliding: SlidingFit) -> None:
        name = self._channel_name()
        series = channel_series(sliding, name, self.settings.alpha_band())
        tracks = track_peaks(sliding, name, self.settings.alpha_band())
        names, medians, spreads, counts = exponent_spread(sliding)
        window = self._clicked_window(sliding, name)
        top = self._top_pair(series)
        if self.every.isChecked() and any(value is not None for value in medians):
            limits = self._shared_limits(medians if self.map_which.currentData() != "spread" else spreads)
            draw_trajectory_with_maps(
                fig, series, tracks, sliding.notes, names, medians, spreads, window, self._inspection,
                top, str(self.map_which.currentData() or "median"), limits,
            )
            self._fill_spread_table(names, medians, spreads, counts, sliding)
        else:
            draw_trajectory(fig, series, tracks, sliding.notes, window, self._inspection, top)
            self.table.setVisible(False)

    def _clicked_window(self, sliding: SlidingFit, name: str):
        if self._inspection is None:
            return None
        center, _span = self._inspection
        fit = window_at(sliding, center)
        if fit is None:
            return (None, None, "No sliding window covers that time. The Spectrum tab is still the fit-bar window.")
        item = next((channel for channel in fit.channels if channel.name == name), None)
        return (fit, item, "")

    def _draw_quality(self, fig) -> None:
        source = self.sources.get(self.condition) or self.recording
        detected = list(self._result) if isinstance(self._result, list) else []
        chosen = []
        for span in self.excluded.get(self.condition, []):
            if span.kind == "user":
                chosen.append(span)
            else:
                chosen.append(QualitySpan(
                    span.channel, span.start_s, span.end_s, "left out", "confirmed", "Chosen on this page.",
                ))
        rows = detected + chosen
        self._quality_rows = rows
        names = [] if source is None else list(source.ch_names)
        draw_quality(fig, names, rows, self._inspection)
        self.table.setVisible(True)
        self.table.blockSignals(True)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Channel", "Start", "End", "Kind", "Level"])
        self.table.setRowCount(len(rows))
        for index, span in enumerate(rows):
            values = [
                span.channel or "all",
                f"{span.start_s:.2f}",
                f"{span.end_s:.2f}",
                span.kind if span.kind != "user" else "drawn",
                span.level,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(index, column, item)
        self.table.blockSignals(False)
        self._arm_span(fig)

    def _write_caption(self) -> None:
        text = self._caption_body()
        if self._inspection is not None and self._mode in {"spectrogram", "trajectory", "raster", "quality", "reactivity", "coherence"}:
            center, span = self._inspection
            if not (self._mode == "reactivity" and isinstance(self._result, dict) and self._result.get("kind") != "single"):
                start = max(0.0, center - span / 2.0)
                text = (
                    f"{text}  Clicked {center:.1f} s. "
                    f"Set fit window writes {start:.1f}–{start + span:.1f} s into the fit bar."
                )
                text += self._clicked_bursts(start, start + span)
                text += self._clicked_column(start, start + span)
                text += self._window_sentence()
        self.caption.setText(text)

    def _caption_body(self) -> str:
        mode = self._mode
        band = band_phrase(*self.settings.alpha_band())
        if mode == "spectrogram" and isinstance(self._result, Spectrogram):
            spec = self._result
            bursts = len(self._episodes())
            burst_text = f"{bursts} burst episode{'s' if bursts != 1 else ''} overlaid." if self.bursts.isChecked() else "Burst overlay is off."
            locked = ""
            if self._color_limits is not None:
                locked = (
                    f" Color locked at {self.color_low.value():.0f} to {self.color_high.value():.0f} dB."
                )
            return f"{spec.channel}. {spec.notes} Color is dB relative to 1 µV²/Hz.{locked} {burst_text}"
        if mode == "trajectory" and isinstance(self._result, SlidingFit):
            sliding = self._result
            extra = f" {sliding.notes}" if sliding.notes else ""
            scope = "Every channel." if self.every.isChecked() else "This channel. The map needs every channel."
            other_key = "eo" if self.condition == "ec" else "ec"
            if self.every.isChecked() and isinstance(self._slidings.get(other_key), SlidingFit):
                scope += " The color scale includes the other open file."
            tracked = self._track_sentence(sliding)
            return (
                f"{self._channel_name()}, {sliding.window_s:.0f} s windows, {sliding.overlap:.0%} overlap, "
                f"{self.settings.aperiodic_mode} slope, {band}.{extra} "
                f"The top trace is {self.top_trace.currentText().lower()}. "
                f"Welch segment {self.settings.segment_s:g} s. "
                f"A missing peak is a gap. An open circle is a weak fit or a peak at the narrowest allowed width. "
                f"{scope}{tracked}"
            )
        if mode == "raster":
            return "Each bar is one burst episode. Channels that light up together are not thereby connected."
        if mode == "durations" and isinstance(self.cycle_fit, CycleFit):
            text = duration_caption(self.cycle_fit, burst_stats(self.cycle_fit))
            if self._partner_fit() is not None:
                text += " Both recordings use the same bins."
            return text
        if mode == "shape" and isinstance(self.cycle_fit, CycleFit):
            how = "Normalized trough to trough." if self.normalized.isChecked() else (
                "Aligned on the peak." if self.align.currentData() == "peak" else "Aligned on the trough."
            )
            return (
                f"{self.cycle_fit.channel}. {how} The band is the median. The shade is the 10th to 90th percentile. "
                "The histograms are amplitude, period, rise/decay, and peak/trough."
            )
        if mode == "scatter" and isinstance(self.cycle_fit, CycleFit):
            return (
                f"{self.cycle_fit.channel}. Frequency comes from the cycle period. "
                "The line only describes the points on screen. It is not a fitted law."
            )
        if mode == "reactivity" and isinstance(self._result, dict):
            return str(self._result.get("note") or self._result.get("title") or "") + self._comparison_sentence()
        if mode == "coherence":
            current, _other, _label = self._coherence_parts()
            if current is None:
                return self._cycle_wait()
            result = current
            zero = ""
            from ..analysis.coherence import zero_lag_mask
            if zero_lag_mask(result).any():
                zero = (
                    " Orange marks zero-lag coupling, which volume conduction can produce."
                )
            thin = f" {result.note}" if result.note else ""
            return (
                f"{result.channel_a} and {result.channel_b}, {result.reference} reference, "
                f"{result.n_segments} segments, {result.accepted_s:.0f} s accepted. "
                f"High coherence is not a connection between regions.{zero}{thin}"
            )
        if mode == "coupling" and self._result is not None and hasattr(self._result, "note"):
            result = self._result
            warning = " ".join(result.warnings)
            note = result.note
            return " ".join(part for part in (warning, note) if part) or "Coupling estimate."
        if mode == "quality":
            return (
                "Red is detected, orange is suspected, blue was already left out, teal was chosen here. "
                "Leave out keeps a stretch out of the next fit. The file is not rewritten."
            )
        return self._cycle_wait()

    def _on_click(self, event) -> None:
        if event.xdata is None or self.recording is None:
            return
        self._press = (float(event.xdata), None if event.ydata is None else float(event.ydata))
        if self._mode == "coherence":
            if event.inaxes is None or event.inaxes.get_label() != "raw":
                return
        elif self._mode not in {"spectrogram", "trajectory", "raster", "quality", "reactivity"}:
            return
        if self._mode == "reactivity" and isinstance(self._result, dict) and self._result.get("kind") != "single":
            return
        center = float(event.xdata)
        span = self._span_at(center)
        self._inspection = (center, span)
        self.inspect_at.emit(center, span)
        self._draw()

    def _span_at(self, center: float) -> float:
        if self._mode == "spectrogram" and isinstance(self._result, Spectrogram):
            span = self._result.window_s
        elif self._mode == "trajectory" and isinstance(self._result, SlidingFit):
            span = self._result.window_s
        else:
            span = 4.0
        if self._mode == "raster":
            for fit in self._raster_fits():
                for episode in fit.episodes:
                    if episode.start_s <= center <= episode.end_s:
                        return max(episode.end_s - episode.start_s, 0.5)
        return span

    def _stage(self) -> None:
        if self._inspection is None:
            self.status.emit("Click a time on the plot first.")
            return
        center, span = self._inspection
        start = max(0.0, center - span / 2.0)
        self.window_requested.emit(start, max(span, 1.0))

    def _leave_out(self) -> None:
        span = self._chosen_span()
        if span is None or span.kind in {"left out", "user"}:
            self.status.emit("Select a detected row first.")
            return
        channel = span.channel or "*"
        items = self.excluded.setdefault(self.condition, [])
        if any(
            item.channel == channel
            and abs(item.start_s - span.start_s) < 0.05
            and abs(item.end_s - span.end_s) < 0.05
            for item in items
        ):
            self.status.emit("That stretch is already left out.")
            return
        items.append(QualitySpan(channel, span.start_s, span.end_s, span.kind, "confirmed", span.note))
        self._emit_mask()

    def _restore(self) -> None:
        span = self._chosen_span()
        items = self.excluded.get(self.condition, [])
        if span is None or not items:
            self.status.emit("Select a row that was left out on this page.")
            return
        channel = span.channel or "*"
        kept = [
            item for item in items
            if not (
                item.channel == channel
                and abs(item.start_s - span.start_s) < 0.05
                and abs(item.end_s - span.end_s) < 0.05
            )
        ]
        if len(kept) == len(items):
            self.status.emit("Select a row that was left out on this page.")
            return
        self.excluded[self.condition] = kept
        self._emit_mask()

    def _emit_mask(self) -> None:
        source = self.sources.get(self.condition) or self.recording
        if source is None:
            return
        self.exclusions_changed.emit(mask_spans(source, self.excluded.get(self.condition, [])))

    def _chosen_span(self) -> QualitySpan | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        index = rows[0].row()
        if not 0 <= index < len(self._quality_rows):
            return None
        return self._quality_rows[index]

    def _local_mode(self) -> bool:
        if self._mode == "raster" and not self.every.isChecked():
            picked = self._picked_indexes()
            return len(picked) <= 1 and (not picked or picked[0] == self._index())
        return self._mode in {"shape", "durations", "scatter"}

    def _episodes(self):
        fit = self.cycle_fit
        if not self.bursts.isChecked() or fit is None or fit.channel != self._channel_name():
            return []
        return list(fit.episodes)

    def _overlay_tracks(self):
        sliding = self._sliding
        if sliding is None or self._sliding_id is None:
            return []
        current = self._cache_key()
        if not self._same_recording(current):
            return []
        return track_peaks(sliding, self._channel_name(), self.settings.alpha_band())

    def _raster_fits(self) -> list[CycleFit]:
        if isinstance(self._result, list):
            return [item for item in self._result if isinstance(item, CycleFit)]
        if isinstance(self.cycle_fit, CycleFit) and self.cycle_fit.channel == self._channel_name():
            return [self.cycle_fit]
        return []

    def _raw_stretch(self):
        recording = self.recording
        if recording is None or self._inspection is None:
            return None
        center, span = self._inspection
        half = max(span, 2.0)
        i0 = max(0, int(round((center - half) * recording.sfreq)))
        i1 = min(recording.n_samples, int(round((center + half) * recording.sfreq)))
        if i1 <= i0:
            return None
        times = np.arange(i0, i1) / recording.sfreq
        values = np.asarray(recording.data[self._index(), i0:i1], dtype=np.float64)
        accepted = recording.mask(self._index())[i0:i1] & np.isfinite(values)
        return times, values, accepted

    def _on_color_lock(self, checked: bool) -> None:
        if not checked or not isinstance(self._result, Spectrogram):
            self._color_limits = None
        else:
            finite = self._result.power_db[np.isfinite(self._result.power_db)]
            if finite.size == 0:
                self._color_limits = None
            else:
                low, high = np.percentile(finite, [2, 98])
                self.color_low.blockSignals(True)
                self.color_high.blockSignals(True)
                self.color_low.setValue(float(low))
                self.color_high.setValue(float(high))
                self.color_low.blockSignals(False)
                self.color_high.blockSignals(False)
                self._color_limits = (float(self.color_low.value()), float(self.color_high.value()))
        self._apply_visibility()
        if isinstance(self._result, Spectrogram):
            self._draw()

    def _clicked_bursts(self, start: float, end: float) -> str:
        hits = [
            episode for episode in self._episodes()
            if episode.end_s >= start and episode.start_s <= end
        ]
        if not hits:
            return ""
        shown = hits[:3]
        parts = []
        for episode in shown:
            duration = episode.end_s - episode.start_s
            freq = ""
            if episode.period_ms is not None and episode.period_ms > 0:
                freq = f", {1000.0 / episode.period_ms:.1f} Hz"
            amp = "" if episode.volt_amp is None else f", {episode.volt_amp:.1f} µV"
            parts.append(
                f"{episode.start_s:.1f}–{episode.end_s:.1f} s, {duration:.2f} s, "
                f"{episode.n_cycles} cycles{amp}{freq}"
            )
        extra = len(hits) - len(shown)
        tail = f", and {extra} more" if extra else ""
        return " Bursts in that stretch: " + "; ".join(parts) + tail + "."

    def _window_sentence(self) -> str:
        if self._mode != "trajectory" or not isinstance(self._result, SlidingFit) or self._inspection is None:
            return ""
        center, _span = self._inspection
        fit = window_at(self._result, center)
        if fit is None:
            return " No sliding window covers that time. The Spectrum tab is still the fit-bar window."
        item = next((channel for channel in fit.channels if channel.name == self._channel_name()), None)
        if item is None or item.spectrum is None:
            reason = item.reason if item is not None and item.reason else "No fit for this channel."
            return f" {fit.t_start:.0f}–{fit.t_end:.0f} s. {reason}"
        peak = alpha_peak(item, self.settings.alpha_band())
        exponent = "no exponent" if item.exponent is None else f"exponent {item.exponent:.2f}"
        peak_text = "no in-band peak" if peak is None else f"in-band peak {peak[0]:.1f} Hz"
        return f" Window {fit.t_start:.0f}–{fit.t_end:.0f} s, {exponent}, {peak_text}."

    def control_values(self) -> dict:
        """Spectrogram and sliding-window settings, for the session metadata."""
        values = {"time_view": self._mode}
        result = self._result
        if isinstance(result, Spectrogram):
            values.update({
                "spectrogram_window_s": result.window_s,
                "spectrogram_overlap": result.overlap,
                "spectrogram_df_hz": result.df_hz,
                "spectrogram_dt_s": result.dt_s,
            })
        if self._color_limits is not None:
            values["spectrogram_color_db"] = [self._color_limits[0], self._color_limits[1]]
        if isinstance(self._sliding, SlidingFit):
            values["sliding_window_s"] = self._sliding.window_s
            values["sliding_overlap"] = self._sliding.overlap
        return values

    def _options(self) -> dict:
        source = self.sources.get(self.condition) or self.recording
        return {
            "window_s": self._window_s(),
            "overlap": self._overlap(),
            "every": self.every.isChecked(),
            "band": self.settings.alpha_band(),
            "other": self._other_index(),
            "phase": (float(self.phase_low.value()), float(self.phase_high.value())),
            "amp": (float(self.amp_low.value()), float(self.amp_high.value())),
            "map": self.comod.isChecked(),
            "continuous": self.continuous,
            "pair": self.recordings,
            "partner": self._partner_recording(),
            "partner_label": self._partner_label(),
            "pre_s": float(self.react_pre.value()),
            "post_s": float(self.react_post.value()),
            "indexes": self._picked_indexes(),
            "amp_index": self._amp_index(),
            "nonsinusoidal": self._nonsinusoidal(),
            "left_label": self.pair_names[0],
            "right_label": self.pair_names[1],
            "source": source,
            "channel": self._job_channel(),
        }

    def _window_s(self) -> float:
        widget = self.spec_window if self._mode == "spectrogram" else self.slide_window
        return float(widget.currentData())

    def _overlap(self) -> float:
        widget = self.spec_overlap if self._mode == "spectrogram" else self.slide_overlap
        return float(widget.currentData())

    def _fit_settings(self) -> FitSettings:
        recording = self.recording
        if recording is None:
            return self.settings
        others = [item for item in self.recordings.values() if item is not recording]
        return matched_settings(self.settings, recording, others)

    def _cache_key(self) -> tuple:
        recording = self.recording
        ident = None if recording is None else (
            recording.source_path, recording.n_samples, float(recording.sfreq), id(recording),
        )
        partner = self._partner_recording()
        partner_id = None if partner is None else (
            partner.source_path, partner.n_samples, float(partner.sfreq), id(partner),
        )
        if self._mode == "quality":
            source = self.sources.get(self.condition) or recording
            ident = None if source is None else ("quality", source.source_path, id(source))
        return (
            self._mode,
            ident,
            partner_id,
            self._index(),
            self.settings.key(),
            self._window_s(),
            self._overlap(),
            self.every.isChecked(),
            self._other_index(),
            round(float(self.phase_low.value()), 2),
            round(float(self.phase_high.value()), 2),
            round(float(self.amp_low.value()), 2),
            round(float(self.amp_high.value()), 2),
            self.comod.isChecked(),
            id(self.continuous),
            tuple(self._picked_indexes()),
            round(float(self.react_pre.value()), 2),
            round(float(self.react_post.value()), 2),
            self._react_index(),
            self._amp_index(),
            self._nonsinusoidal(),
        )

    def _index(self) -> int:
        if self.recording is None or self.recording.n_channels == 0:
            return 0
        return min(max(self._selected, 0), self.recording.n_channels - 1)

    def _channel_name(self) -> str:
        recording = self.recording
        if recording is None or not recording.ch_names:
            return ""
        return recording.ch_names[self._index()]

    def _other_index(self) -> int:
        if self.other.count() == 0:
            return 0
        data = self.other.currentData()
        return 0 if data is None else int(data)

    def _fill_other(self) -> None:
        recording = self.recording
        if recording is None:
            return
        current = self.other.currentData()
        self.other.blockSignals(True)
        self.other.clear()
        for index, name in enumerate(recording.ch_names):
            if index == self._index():
                continue
            self.other.addItem(name, index)
        if self.other.count() == 0 and recording.ch_names:
            self.other.addItem(recording.ch_names[0], 0)
        restore = self.other.findData(current)
        if restore >= 0:
            self.other.setCurrentIndex(restore)
        elif self.other.count():
            preferred = _posterior_index(list(recording.ch_names), self._index())
            found = self.other.findData(preferred)
            if found >= 0:
                self.other.setCurrentIndex(found)
        self.other.blockSignals(False)
        self._fill_amp()

    def _apply_visibility(self) -> None:
        mode = self._mode
        self.spec_window.setVisible(mode == "spectrogram")
        self.spec_overlap.setVisible(mode == "spectrogram")
        self.lock_color.setVisible(mode == "spectrogram")
        locked = mode == "spectrogram" and self.lock_color.isChecked()
        self.color_low.setVisible(locked)
        self.color_high.setVisible(locked)
        self.top_trace.setVisible(mode == "trajectory")
        self.map_which.setVisible(mode == "trajectory" and self.every.isChecked())
        self.slide_window.setVisible(mode == "trajectory")
        self.slide_overlap.setVisible(mode == "trajectory")
        self.bursts.setVisible(mode in {"spectrogram", "shape", "scatter"})
        self.every.setVisible(mode in {"trajectory", "raster"})
        self.picks.setVisible(mode == "raster" and not self.every.isChecked())
        self.align.setVisible(mode == "shape")
        self.normalized.setVisible(mode == "shape")
        self.other.setVisible(mode == "coherence")
        self.amp_channel.setVisible(mode == "coupling")
        react = mode == "reactivity"
        self.react_pre.setVisible(react)
        self.react_post.setVisible(react)
        self.react_channel.setVisible(react)
        for widget in (self.phase_low, self.phase_high, self.amp_low, self.amp_high, self.comod):
            widget.setVisible(mode == "coupling")
        quality = mode == "quality"
        self.leave.setVisible(quality)
        self.restore.setVisible(quality)
        self.table.setVisible(quality)
        self.stage.setVisible(mode in {"spectrogram", "trajectory", "raster", "quality", "reactivity"})

    def _waiting(self) -> str:
        name = self._channel_name() or "the channel"
        return {
            "spectrogram": f"Computing the spectrogram of {name}…",
            "trajectory": f"Fitting {name} window by window…",
            "raster": "Finding bursts…",
            "reactivity": "Computing band power…",
            "coherence": "Computing coherence…",
            "coupling": "Computing coupling…",
            "quality": "Checking the recording…",
        }.get(self._mode, "Working…")

    def _cycle_wait(self) -> str:
        if self.cycle_fit is None:
            return "The cycle fit is still running."
        if self.cycle_fit.channel != self._channel_name():
            return f"The cycle fit is for {self.cycle_fit.channel}. This channel is still running."
        return "No cycles in this fit."

    def _settings_key(self) -> str:
        return repr(self.settings.key())

    def _picked_indexes(self) -> list[int]:
        if self.recording is None:
            return [0]
        if self.every.isChecked() or self.picks.count() == 0:
            if self.every.isChecked():
                return list(range(self.recording.n_channels))
            return [self._index()]
        picked = []
        for row in range(self.picks.count()):
            item = self.picks.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                picked.append(int(item.data(Qt.UserRole)))
        return picked or [self._index()]

    def _fill_picks(self) -> None:
        recording = self.recording
        if recording is None:
            return
        names = list(recording.ch_names)
        current = [self.picks.item(index).text() for index in range(self.picks.count())]
        if current == names:
            return
        self.picks.blockSignals(True)
        self.picks.clear()
        for index, name in enumerate(names):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(Qt.UserRole, index)
            item.setCheckState(Qt.Checked if index == self._index() else Qt.Unchecked)
            self.picks.addItem(item)
        self.picks.blockSignals(False)

    def _fill_react_channel(self) -> None:
        recording = self.recording
        if recording is None:
            return
        current = self._channel_name()
        self.react_channel.blockSignals(True)
        self.react_channel.clear()
        for index, name in enumerate(recording.ch_names):
            self.react_channel.addItem(name, index)
        found = self.react_channel.findText(current)
        if found >= 0:
            self.react_channel.setCurrentIndex(found)
        self.react_channel.blockSignals(False)

    def _fill_amp(self) -> None:
        recording = self.recording
        if recording is None:
            return
        current = self.amp_channel.currentData()
        self.amp_channel.blockSignals(True)
        self.amp_channel.clear()
        self.amp_channel.addItem("This channel", -1)
        for index, name in enumerate(recording.ch_names):
            if index == self._index():
                continue
            self.amp_channel.addItem(name, index)
        restore = self.amp_channel.findData(current)
        self.amp_channel.setCurrentIndex(restore if restore >= 0 else 0)
        self.amp_channel.blockSignals(False)

    def _react_changed(self) -> None:
        name = self.react_channel.currentText()
        if name and name != self._channel_name():
            self.channel_requested.emit(name)

    def _react_index(self) -> int:
        data = self.react_channel.currentData()
        return self._index() if data is None else int(data)

    def _amp_index(self) -> int:
        data = self.amp_channel.currentData()
        if data is None or int(data) < 0:
            return -1
        return int(data)

    def _job_channel(self) -> str:
        if self._mode == "reactivity" and self.react_channel.currentText():
            return self.react_channel.currentText()
        return self._channel_name()

    def _offer_posterior(self) -> bool:
        recording = self.recording
        if recording is None or self._posterior_for == id(recording):
            return False
        self._posterior_for = id(recording)
        if self._channel_name() in {"O1", "O2", "Oz", "Pz"}:
            return False
        for name in ("O1", "O2", "Oz", "Pz"):
            if name in recording.ch_names:
                self.channel_requested.emit(name)
                return True
        return False

    def _nonsinusoidal(self) -> bool:
        fit = self.cycle_fit
        if fit is None or fit.channel != self._channel_name():
            return False
        for value in (fit.rise_decay, fit.peak_trough):
            if value is not None and abs(float(value) - 0.5) >= 0.15:
                return True
        return False

    def _partner_recording(self):
        other = "eo" if self.condition == "ec" else "ec"
        partner = self.recordings.get(other)
        if partner is None or partner is self.recording:
            return None
        return partner

    def _partner_label(self) -> str:
        return self.pair_names[1] if self.condition == "ec" else self.pair_names[0]

    def _partner_fit(self) -> CycleFit | None:
        fit = self.cycle_fit
        partner = self.pair_cycles.get("eo" if self.condition == "ec" else "ec")
        if fit is None or not isinstance(partner, CycleFit) or partner.channel != fit.channel:
            return None
        return partner

    def _duration_groups(self):
        fit = self.cycle_fit
        partner = self._partner_fit()
        if fit is None:
            return []
        mine = self.pair_names[0] if self.condition == "ec" else self.pair_names[1]
        theirs = self.pair_names[1] if self.condition == "ec" else self.pair_names[0]
        groups = [(mine if partner is not None else fit.channel, burst_stats(fit).durations_s)]
        if partner is not None:
            groups.append((theirs, burst_stats(partner).durations_s))
        return groups

    def _scatter_groups(self):
        fit = self.cycle_fit
        if fit is None:
            return []
        partner = self._partner_fit()
        if partner is None:
            return [(fit.channel, fit, "#1c4e78")]
        mine = self.pair_names[0] if self.condition == "ec" else self.pair_names[1]
        theirs = self.pair_names[1] if self.condition == "ec" else self.pair_names[0]
        return [
            (mine, fit, "#1c4e78"),
            (theirs, partner, "#c2410c"),
        ]

    def _coherence_parts(self):
        result = self._result
        if isinstance(result, dict) and "this" in result:
            return result.get("this"), result.get("other"), self._partner_label()
        if result is not None and hasattr(result, "coherence"):
            return result, None, ""
        return None, None, ""

    def _coherence_raw(self):
        if self.recording is None:
            return []
        if self._inspection is None:
            return [("empty", self.recording.duration_s)]
        return [self._stretch(self._index()), self._stretch(self._other_index())]

    def _stretch(self, index: int):
        recording = self.recording
        center, span = self._inspection
        half = max(span, 2.0)
        i0 = max(0, int(round((center - half) * recording.sfreq)))
        i1 = min(recording.n_samples, int(round((center + half) * recording.sfreq)))
        times = np.arange(i0, i1) / recording.sfreq
        values = np.asarray(recording.data[index, i0:i1], dtype=np.float64)
        accepted = recording.mask(index)[i0:i1] & np.isfinite(values)
        name = recording.ch_names[index] if index < len(recording.ch_names) else ""
        return times, values, accepted, name

    def _top_pair(self, series):
        which = self.top_trace.currentData()
        if which == "offset":
            return series.offset, "Offset"
        if which == "knee":
            knee = series.knee if series.knee is not None else np.full(series.times_s.shape, np.nan)
            return knee, "Knee"
        return series.exponent, "Exponent"

    def _shared_limits(self, values):
        other = self._partner if isinstance(self._partner, SlidingFit) else None
        if other is None:
            other = self._slidings.get("eo" if self.condition == "ec" else "ec")
        if not isinstance(other, SlidingFit):
            return None
        _names, medians, spreads, _counts = exponent_spread(other)
        chosen = spreads if self.map_which.currentData() == "spread" else medians
        pool = [
            value for value in list(values) + list(chosen)
            if value is not None and np.isfinite(value)
        ]
        if len(pool) < 2:
            return None
        low, high = float(min(pool)), float(max(pool))
        if high - low < 1e-9:
            return None
        return low, high

    def _track_sentence(self, sliding: SlidingFit) -> str:
        series = channel_series(sliding, self._channel_name(), self.settings.alpha_band())
        tracks = track_peaks(sliding, series.channel, self.settings.alpha_band())
        present = int(np.isfinite(series.peak_hz).sum())
        text = track_summary(tracks, int(series.times_s.size), present)
        return f" {text}" if text else ""

    def _sliding_current(self):
        sliding = self._sliding
        if sliding is None or self._sliding_id is None:
            return None
        current = self._cache_key()
        if not self._same_recording(current):
            return None
        return sliding

    def _same_recording(self, current: tuple) -> bool:
        """The stored sliding fit belongs to this recording and these fit settings."""
        stored = self._sliding_id
        if stored is None or len(stored) < 5 or len(current) < 5:
            return False
        return stored[1] == current[1] and stored[4] == current[4]

    def _clicked_column(self, start: float, end: float) -> str:
        parts = []
        if self._mode == "spectrogram" and isinstance(self._result, Spectrogram) and self._result.times_s.size:
            spec = self._result
            center = self._inspection[0] if self._inspection is not None else start
            index = int(np.argmin(np.abs(spec.times_s - center)))
            column = spec.power_db[:, index]
            if not np.isfinite(column).any():
                parts.append(f" The column at {spec.times_s[index]:.2f} s is blank.")
            else:
                peak = int(np.nanargmax(column))
                parts.append(
                    f" Column at {spec.times_s[index]:.2f} s, strongest bin "
                    f"{spec.freqs_hz[peak]:.2f} Hz at {column[peak]:.1f} dB."
                )
        parts.append(self._cycles_in_span(start, end))
        return "".join(parts)

    def _cycles_in_span(self, start: float, end: float) -> str:
        fit = self.cycle_fit
        if fit is None or fit.channel != self._channel_name() or not fit.spans or fit.sfreq <= 0:
            return ""
        hits = 0
        bursts = 0
        for first, second, is_burst in fit.spans:
            t0 = fit.t_offset + first / fit.sfreq
            t1 = fit.t_offset + second / fit.sfreq
            if t1 >= start and t0 <= end:
                hits += 1
                bursts += int(bool(is_burst))
        if not hits:
            return ""
        return f" {hits} cycles in that stretch, {bursts} of them in a burst."

    def _comparison_sentence(self) -> str:
        if not isinstance(self._result, dict) or self._result.get("kind") != "separate":
            return ""
        left = self.spectra.get("ec")
        right = self.spectra.get("eo")
        if left is None or right is None:
            return ""
        try:
            rows = compare_pair(left, right, self.settings.alpha_band())
        except (TypeError, ValueError):
            return ""
        row = next((item for item in rows if item.name == self._channel_name()), None)
        if row is None:
            return ""
        parts = []
        if row.alpha_reactivity is not None:
            parts.append(f"peak height Δ {row.alpha_reactivity:+.2f}")
        if row.exponent_delta is not None:
            parts.append(f"exponent Δ {row.exponent_delta:+.2f}")
        if row.band_power_reactivity_percent is not None:
            parts.append(f"Δ band {row.band_power_reactivity_percent:+.0f}%")
        if not parts:
            return ""
        return " " + ", ".join(parts) + "."

    def _fill_episode_table(self, fits) -> None:
        rows = []
        for fit in fits:
            for episode in fit.episodes:
                freq = ""
                if episode.period_ms is not None and episode.period_ms > 0:
                    freq = f"{1000.0 / episode.period_ms:.2f}"
                amp = "" if episode.volt_amp is None else f"{episode.volt_amp:.1f}"
                rows.append((
                    fit.channel,
                    f"{episode.start_s:.2f}",
                    f"{episode.end_s - episode.start_s:.2f}",
                    str(episode.n_cycles),
                    amp,
                    freq,
                ))
        self._show_table(["Channel", "Start", "Duration", "Cycles", "Amplitude", "Frequency"], rows)

    def _fill_spread_table(self, names, _medians, _spreads, counts, sliding: SlidingFit) -> None:
        rows = []
        for name, count in zip(names, counts):
            fraction = self._rejected_fraction(sliding, name)
            shown = "—" if not np.isfinite(fraction) else f"{fraction:.2f}"
            rows.append((name, str(count), shown))
        self._show_table(["Channel", "Usable windows", "Rejected fraction"], rows)

    def _rejected_fraction(self, sliding: SlidingFit, name: str) -> float:
        recording = self.recording
        if recording is None or name not in recording.ch_names or not sliding.windows:
            return float("nan")
        index = recording.ch_names.index(name)
        mask = recording.mask(index)
        fractions = []
        for fit in sliding.windows:
            i0 = max(0, int(round(fit.t_start * recording.sfreq)))
            i1 = min(recording.n_samples, int(round(fit.t_end * recording.sfreq)))
            if i1 <= i0:
                continue
            piece = mask[i0:i1]
            fractions.append(0.0 if piece.size == 0 else 1.0 - float(np.mean(piece)))
        if not fractions:
            return float("nan")
        return float(np.median(fractions))

    def _show_table(self, headers, rows) -> None:
        self.table.setVisible(True)
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_index, column, item)
        self.table.blockSignals(False)

    def _on_color_range(self, _value: float = 0.0) -> None:
        if not self.lock_color.isChecked():
            return
        low, high = float(self.color_low.value()), float(self.color_high.value())
        if high <= low:
            high = low + 1.0
            self.color_high.blockSignals(True)
            self.color_high.setValue(high)
            self.color_high.blockSignals(False)
        self._color_limits = (low, high)
        if isinstance(self._result, Spectrogram):
            self._draw()

    def _redraw_only(self, *_args) -> None:
        self._apply_visibility()
        if self.recording is not None and (self._result is not None or self._local_mode()):
            self._draw()

    def _on_release(self, event) -> None:
        press = self._press
        self._press = None
        if self._mode != "quality" or press is None or event.xdata is None or self.recording is None:
            return
        x0, y0 = press
        x1 = float(event.xdata)
        if abs(x1 - x0) < 0.2:
            return
        span = QualitySpan(
            self._channel_at(y0), min(x0, x1), max(x0, x1),
            "user", "confirmed", "Drawn on this page.",
        )
        self.excluded.setdefault(self.condition, []).append(span)
        self._emit_mask()

    def _channel_at(self, y: float | None) -> str:
        source = self.sources.get(self.condition) or self.recording
        names = [] if source is None else list(source.ch_names)
        if not names or y is None:
            return self._channel_name()
        index = int(round((len(names) - 1) - float(y)))
        return names[min(max(index, 0), len(names) - 1)]

    def _arm_span(self, _fig) -> None:
        return

    def _accept(self, result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], SlidingFit):
            self._partner = result[1] if isinstance(result[1], SlidingFit) else None
            return result[0]
        if self._mode == "trajectory":
            self._partner = None
        return result

    def _stored(self, result):
        if self._mode == "trajectory" and isinstance(result, SlidingFit):
            return (result, self._partner)
        return result

    def _empty(self, message: str) -> None:
        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color="#63748a", fontsize=12)
        self.canvas.draw_idle()


def _job(mode, recording, settings, index, options):
    if mode == "spectrogram":
        return channel_spectrogram(recording, settings, index, options["window_s"], options["overlap"])
    if mode == "trajectory":
        channel = None if options["every"] else index
        primary = sliding_spectra(recording, settings, options["window_s"], options["overlap"], channel)
        partner = None
        other = options.get("partner") if options["every"] else None
        if other is not None:
            try:
                partner = sliding_spectra(other, settings, options["window_s"], options["overlap"], None)
            except ValueError:
                partner = None
        return primary, partner
    if mode == "raster":
        indexes = options.get("indexes") or (
            list(range(recording.n_channels)) if options["every"] else [index]
        )
        fits = []
        for channel in indexes:
            try:
                fits.append(fit_cycles(recording, channel, options["band"], settings))
            except ValueError:
                continue
        return fits
    if mode == "reactivity":
        return _reactivity(recording, settings, index, options)
    if mode == "coherence":
        result = coherence_pair(recording, settings, index, int(options["other"]))
        return {"this": result, "other": _partner_coherence(recording, settings, index, options)}
    if mode == "coupling":
        grid = None
        if options["map"]:
            grid = (np.array([4.0, 6.0, 8.0, 10.0, 12.0]), np.array([20.0, 30.0, 40.0]))
        amplitude = options.get("amp_index")
        if amplitude is None or int(amplitude) < 0:
            amplitude = None
        return phase_amplitude(
            recording, settings, index, options["phase"], options["amp"],
            n_surrogates=200, comodulogram_grid=grid,
            amp_index=amplitude, nonsinusoidal=bool(options.get("nonsinusoidal")),
        )
    if mode == "quality":
        return quality_spans(options["source"], settings.amplitude_limit_uv)
    raise ValueError("That view is not available.")


def _reactivity(recording, settings, index, options) -> dict:
    band = options["band"]
    name = options.get("channel") or recording.ch_names[index]
    continuous = options.get("continuous")
    if continuous is not None and name in continuous.ch_names:
        events = [mark.onset_s for mark in continuous.marks if condition_in_text(mark.label)]
        if events:
            full = replace(settings, t_start=0.0, t_length=None)
            times, power = block_band_power(
                continuous, full, continuous.ch_names.index(name), band, 2.0, 0.5,
            )
            pre = float(options.get("pre_s", 10.0))
            post = float(options.get("post_s", 20.0))
            grid, curves = align_to_events(times, power, events, pre, post)
            change = _event_change(grid, curves)
            extras = []
            short = ""
            if pre + post < 8.0:
                short = " The window is shorter than 8 s, so only band power is shown."
            else:
                courses = _event_parameters(continuous, full, continuous.ch_names.index(name), events, pre, post, band)
                if courses:
                    extras = list(courses)
                    if len(events) > 8:
                        short = " Peak height and exponent use the first 8 marks."
            return {
                "kind": "events",
                "grid": grid,
                "curves": curves,
                "extras": extras,
                "events": events,
                "title": f"{name} around {len(events)} eye mark{'s' if len(events) != 1 else ''}",
                "note": (
                    f"{name}, {band_phrase(*band)}, {pre:.0f} s before and {post:.0f} s after "
                    f"eye marks on the original file. "
                    f"{change} The split sides do not contain the crossing.{short}"
                ),
            }
    pair = options.get("pair") or {}
    if "ec" in pair and "eo" in pair:
        return {
            "kind": "separate",
            "left": _one_power(pair["ec"], settings, name, band),
            "right": _one_power(pair["eo"], settings, name, band),
            "left_label": options.get("left_label") or "Eyes closed",
            "right_label": options.get("right_label") or "Eyes open",
            "note": (
                f"{name}, {band_phrase(*band)}, 2 s blocks. "
                "These are separate recordings. There is no transition between them."
            ),
        }
    times, power = block_band_power(recording, settings, index, band, 2.0, 0.5)
    return {
        "kind": "single",
        "single": (times, power),
        "title": f"{name}   ·   {band_phrase(*band)}   ·   2 s blocks",
        "note": f"{name}, {band_phrase(*band)}, mean Welch power in 2 s blocks.",
    }


def _partner_coherence(recording, settings, index, options):
    partner = options.get("partner")
    if partner is None or index >= len(recording.ch_names):
        return None
    other = int(options.get("other", 0))
    if other >= len(recording.ch_names):
        return None
    left = recording.ch_names[index]
    right = recording.ch_names[other]
    if left not in partner.ch_names or right not in partner.ch_names or left == right:
        return None
    try:
        return coherence_pair(
            partner, settings, partner.ch_names.index(left), partner.ch_names.index(right),
        )
    except ValueError:
        return None


def _event_parameters(recording, settings, index, events, pre, post, band):
    """specparam on each peri-event stretch when it is long enough for an 8 s window.

    Each mark is aligned on its own. Joining the marks first would treat the
    gap between them as the sample step.
    """
    grid = None
    heights = []
    exponents = []
    for event in list(events)[:8]:
        start = max(0.0, float(event) - float(pre))
        stop = min(recording.duration_s, float(event) + float(post))
        if stop - start < 8.0:
            continue
        i0 = int(round(start * recording.sfreq))
        i1 = int(round(stop * recording.sfreq))
        if i1 - i0 < int(8 * recording.sfreq):
            continue
        piece = recording.cropped(i0, i1)
        if index >= piece.n_channels:
            continue
        try:
            sliding = sliding_spectra(piece, settings, 8.0, 0.5, index)
        except ValueError:
            continue
        series = channel_series(sliding, piece.ch_names[index], band)
        origin = i0 / recording.sfreq
        grid_h, curve_h = align_to_events(series.times_s + origin, series.peak_height, [event], pre, post)
        _grid_e, curve_e = align_to_events(series.times_s + origin, series.exponent, [event], pre, post)
        if grid_h.size == 0 or curve_h.size == 0 or curve_e.size == 0:
            continue
        if grid is None:
            grid = grid_h
        if curve_h.shape[-1] != grid.size:
            curve_h = np.interp(grid, grid_h, curve_h[0])[None, :]
            curve_e = np.interp(grid, grid_h, curve_e[0])[None, :]
        heights.append(np.asarray(curve_h[0], dtype=np.float64))
        exponents.append(np.asarray(curve_e[0], dtype=np.float64))
    if grid is None or not heights:
        return None
    return (
        {"grid": grid, "curves": np.vstack(heights), "ylabel": "Peak height (log10)"},
        {"grid": grid, "curves": np.vstack(exponents), "ylabel": "Exponent"},
    )


def _one_power(recording: Recording, settings: FitSettings, name: str, band):
    if name not in recording.ch_names:
        return np.zeros(0), np.zeros(0)
    return block_band_power(recording, settings, recording.ch_names.index(name), band, 2.0, 0.5)


def _event_change(grid, curves) -> str:
    if grid.size == 0 or curves.size == 0:
        return ""
    median = np.full(curves.shape[-1], np.nan, dtype=np.float64)
    for index in range(curves.shape[-1]):
        column = curves[:, index]
        column = column[np.isfinite(column)]
        if column.size:
            median[index] = float(np.median(column))
    before = median[grid < -0.5]
    after = median[(grid >= 0.5) & (grid <= 5.0)]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if before.size == 0 or after.size == 0:
        return ""
    pre = float(np.median(before))
    post = float(np.median(after))
    text = f"Median power {pre:.2f} before the mark and {post:.2f} in the first 5 s after."
    if pre > 1e-6:
        text += f" That is {(post - pre) / pre * 100.0:+.0f}%."
    return text


def _posterior_index(names: list[str], current: int) -> int:
    for name in ("O1", "O2", "Oz", "Pz"):
        if name in names and names.index(name) != current:
            return names.index(name)
    if len(names) > 1:
        return (current + 1) % len(names)
    return current


def _choice(items, index: int) -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    box.setCurrentIndex(index)
    return box


def _hertz(value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.5, 80.0)
    box.setDecimals(1)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setSuffix(" Hz")
    box.setFixedWidth(84)
    return box


def _decibel(tip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-80.0, 80.0)
    box.setDecimals(0)
    box.setSingleStep(1.0)
    box.setSuffix(" dB")
    box.setFixedWidth(78)
    box.setToolTip(tip)
    return box


def _seconds(value: float, tip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(1.0, 120.0)
    box.setDecimals(0)
    box.setSingleStep(1.0)
    box.setValue(value)
    box.setSuffix(" s")
    box.setFixedWidth(72)
    box.setToolTip(tip)
    return box


def _cell(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
