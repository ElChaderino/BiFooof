"""Controls for the time window, spectrum range, and burst rule."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..analysis.cycles import BANDS
from ..analysis.settings import BURST_PRESETS, RANGE_PRESETS, SEGMENT_PRESETS, TIME_PRESETS, FitSettings
from ..analysis.specparam_fit import EPOCH_PEAKS
from .layouts import FlowLayout, control_group, CompactDoubleSpinBox


class FitBar(QFrame):
    """The choices resting-state specparam and bycycle work actually changes."""

    apply_requested = Signal(object)
    band_changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setObjectName("Header")
        self._filling = False
        self._applied = FitSettings()
        self._recording_available = False
        self._busy = False

        self.time = QComboBox()
        for label, _start, _length in TIME_PRESETS:
            self.time.addItem(label)
        self.time.setToolTip(
            "Which stretch of the recording is fit. The trace scrollbar still moves through the whole file."
        )
        self.t_start = _seconds(0.0)
        self.t_length = _seconds(60.0)
        self.t_start.setToolTip("Start of the fit window, in seconds from the beginning of the file.")
        self.t_length.setToolTip("Length of the fit window. Ignored when the whole recording is selected.")

        self.range = QComboBox()
        for label, *_rest in RANGE_PRESETS:
            self.range.addItem(label)
        self.range.setToolTip(
            "1–40 Hz is the broad resting fit. 2–40 Hz is the published eyes-open/eyes-closed range. "
            "3–30 Hz stops before the muscle range, which otherwise pulls the slope. "
            "2–20 Hz keeps theta, alpha, and beta."
        )
        self.fmin = _hertz(1.0)
        self.fmax = _hertz(40.0)
        self.fmin.setToolTip("Lowest frequency in the specparam fit.")
        self.fmax.setToolTip(
            "Highest frequency in the specparam fit. The slope is fit across this whole range."
        )

        self.segment = QComboBox()
        for label, _seconds_value in SEGMENT_PRESETS:
            self.segment.addItem(label, _seconds_value)
        self.segment.setCurrentIndex(1)
        self.segment.setToolTip(
            "Welch segment length. 4 s gives 0.25 Hz bins. Longer segments make a smoother spectrum."
        )

        self.aperiodic = QComboBox()
        self.aperiodic.addItem("Fixed slope", "fixed")
        self.aperiodic.addItem("Knee", "knee")
        self.aperiodic.setToolTip(
            "Fixed is the straight 1/f slope used for 1–40 Hz resting EEG. "
            "Knee lets the slope bend, which shows up when the fit reaches lower or much higher frequencies."
        )

        self.width_low = _hertz(0.5)
        self.width_high = _hertz(12.0)
        self.width_low.setToolTip("Narrowest peak specparam is allowed to call a peak, in Hz.")
        self.width_high.setToolTip("Widest peak specparam is allowed to call a peak, in Hz.")
        self.max_peaks = QSpinBox()
        self.max_peaks.setRange(1, 12)
        self.max_peaks.setValue(6)
        self.max_peaks.setFixedWidth(64)
        self.max_peaks.setToolTip("Maximum number of peaks on one channel.")
        self.min_height = CompactDoubleSpinBox()
        self.min_height.setRange(0.0, 2.0)
        self.min_height.setDecimals(2)
        self.min_height.setSingleStep(0.05)
        self.min_height.setValue(0.05)
        self.min_height.setToolTip("Minimum peak height above the aperiodic fit, in log10 power.")
        self.peak_threshold = CompactDoubleSpinBox()
        self.peak_threshold.setRange(0.5, 6.0)
        self.peak_threshold.setDecimals(1)
        self.peak_threshold.setSingleStep(0.5)
        self.peak_threshold.setValue(2.0)
        self.peak_threshold.setFixedWidth(68)
        self.peak_threshold.setToolTip(
            "How many standard deviations a peak must stand above the spectrum to count. Papers leave this at 2."
        )
        self.band = QComboBox()
        for label, limits in BANDS:
            self.band.addItem(label, limits)
        self.band.addItem("Custom", None)
        self.band.setCurrentIndex(2)
        self.band.setToolTip(
            "Band for the reactivity peak and for bycycle. Alpha 8–12 Hz is the usual eyes-closed band. "
            "Delta, theta, beta, gamma, or a custom range use the same views."
        )
        self.alpha_low = _hertz(8.0)
        self.alpha_high = _hertz(12.0)
        self.alpha_low.setToolTip("Low edge of the selected band, in Hz. The same range is used for reactivity and for bycycle.")
        self.alpha_high.setToolTip("High edge of the selected band, in Hz. The same range is used for reactivity and for bycycle.")

        self.bursts = QComboBox()
        for label, *_rest in BURST_PRESETS:
            self.bursts.addItem(label)
        self.bursts.setToolTip(
            "Library keeps every consistent run of 3 cycles. "
            "Resting alpha and paper alpha are amplitude rules, not a frequency range. "
            "They still apply to the band you select, and they require a taller amplitude before a run counts as a burst."
        )
        self.bursts.addItem("Custom")
        self._custom_burst = []
        for value in (0.0, 0.5, 0.5, 0.8):
            box = CompactDoubleSpinBox()
            box.setRange(0, 1)
            box.setDecimals(3)
            box.setSingleStep(.05)
            box.setValue(value)
            box.valueChanged.connect(self._mark_pending)
            self._custom_burst.append(box)
        self.burst_cycles = QSpinBox()
        self.burst_cycles.setRange(1, 10000)
        self.burst_cycles.setValue(3)
        self.burst_cycles.valueChanged.connect(self._mark_pending)

        self.reference = QComboBox()
        self.reference.addItem("As stored", "stored")
        self.reference.addItem("Average", "average")
        self.reference.addItem("Linked ears", "linked")
        self.reference.setToolTip(
            "As stored keeps the reference in the file. Average subtracts the mean of the fitted channels. "
            "Linked ears uses A1 and A2, or M1 and M2, when the file has them. Press Refit to apply."
        )
        self.epochs = QComboBox()
        self.epochs.addItem("Whole window", None)
        self.epochs.addItem("30 s epochs", 30.0)
        self.epochs.addItem("60 s epochs", 60.0)
        self.epochs.setToolTip(
            "Whole window fits one spectrum. "
            + EPOCH_PEAKS
            + " When both files are open and the time window is the whole recording, both are fit to the shorter length. "
            "Press Refit to apply."
        )
        self.amplitude = CompactDoubleSpinBox()
        self.amplitude.setRange(0.0, 5000.0)
        self.amplitude.setDecimals(0)
        self.amplitude.setSingleStep(50.0)
        self.amplitude.setValue(500.0)
        self.amplitude.setFixedWidth(88)
        self.amplitude.setToolTip(
            "Samples beyond this many microvolts stay out of the spectrum and the cycles. 0 keeps every sample. "
            "Press Refit to apply."
        )

        self.apply = QPushButton("Refit")
        self.apply.setObjectName("Secondary")
        self.apply.setToolTip(
            "Run specparam and bycycle again. This applies the time window, spectrum range, "
            "reference, epochs, amplitude limit, and burst rule. The band updates on its own."
        )
        self.apply.clicked.connect(self._emit)

        for combo in (
            self.time, self.range, self.segment, self.aperiodic,
            self.bursts, self.reference, self.epochs,
        ):
            combo.currentIndexChanged.connect(self._mark_pending)
        for box in (
            self.t_start, self.t_length, self.fmin, self.fmax, self.width_low, self.width_high,
            self.max_peaks, self.min_height, self.peak_threshold, self.amplitude,
        ):
            box.valueChanged.connect(self._mark_pending)

        self.time.currentIndexChanged.connect(self._on_time)
        self.range.currentIndexChanged.connect(self._on_range)
        self.segment.currentIndexChanged.connect(self._mark_custom_range)
        self.aperiodic.currentIndexChanged.connect(self._mark_custom_range)
        for box in (self.t_start, self.t_length):
            box.valueChanged.connect(self._mark_custom_time)
        for box in (self.fmin, self.fmax, self.width_low, self.width_high, self.min_height, self.max_peaks):
            box.valueChanged.connect(self._mark_custom_range)
        self.peak_threshold.valueChanged.connect(self._mark_custom_range)
        self.band.currentIndexChanged.connect(self._on_band)
        self.alpha_low.valueChanged.connect(self._on_band_edge)
        self.alpha_high.valueChanged.connect(self._on_band_edge)
        self._on_time()

        common = QWidget()
        common.setObjectName("ControlGroup")
        flow = FlowLayout(common, 14)
        flow.addWidget(control_group("WINDOW", self.time, _label("START"), self.t_start, _label("LENGTH"), self.t_length))
        flow.addWidget(control_group("SPECTRUM", self.range))
        flow.addWidget(control_group("BAND", self.band, self.alpha_low, self.alpha_high))
        flow.addWidget(control_group("REFERENCE", self.reference))
        flow.addWidget(control_group("EPOCHS", self.epochs))
        flow.addWidget(control_group("LIMIT µV", self.amplitude))
        self.details = QPushButton("More settings")
        self.details.setObjectName("Secondary")
        self.details.setCheckable(True)
        self.details.setToolTip("Frequency limits, Welch segment, aperiodic model, peak limits, and burst rules.")
        actions = QVBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self.apply)
        actions.addWidget(self.details)
        actions.addStretch(1)
        top = QHBoxLayout()
        top.setSpacing(16)
        top.addWidget(common, 1)
        top.addLayout(actions)

        self.advanced = QWidget()
        self.advanced.setObjectName("ControlGroup")
        advanced = FlowLayout(self.advanced, 14)
        advanced.addWidget(control_group("FIT Hz", self.fmin, self.fmax))
        advanced.addWidget(control_group("WELCH", self.segment))
        advanced.addWidget(control_group("MODEL", self.aperiodic))
        advanced.addWidget(control_group("WIDTH Hz", self.width_low, self.width_high))
        advanced.addWidget(control_group("MAX PEAKS", self.max_peaks))
        advanced.addWidget(control_group("MIN HEIGHT", self.min_height))
        advanced.addWidget(control_group("THRESHOLD SD", self.peak_threshold))
        advanced.addWidget(control_group("BURSTS", self.bursts))
        self.burst_options = control_group("CUSTOM BURST: amp / amp consistency / period / monotonicity / cycles", *self._custom_burst, self.burst_cycles)
        advanced.addWidget(self.burst_options)
        self.burst_options.hide()
        self.bursts.currentIndexChanged.connect(lambda: self.burst_options.setVisible(self.bursts.currentText() == "Custom"))
        self.advanced.hide()
        self.details.toggled.connect(self._toggle_details)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        self.feedback.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 8, 16, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.advanced)
        layout.addWidget(self.feedback)
        self._refresh_feedback()

    def _toggle_details(self, checked: bool) -> None:
        self.advanced.setVisible(checked)
        self.details.setText("Hide settings" if checked else "More settings")

    def set_context(self, recording_available: bool, busy: bool = False) -> None:
        if (recording_available, busy) == (self._recording_available, self._busy):
            return
        self._recording_available = recording_available
        self._busy = busy
        self._refresh_feedback()

    def mark_applied(self, settings: FitSettings) -> None:
        self._applied = replace(settings)
        self._refresh_feedback()

    def _refresh_feedback(self) -> None:
        if not hasattr(self, "feedback"):
            return
        chosen = self.settings()
        try:
            chosen.validate()
        except ValueError as exc:
            self.feedback.setText(str(exc))
            self.feedback.setObjectName("ValidationError")
            self.feedback.show()
            self.apply.setEnabled(False)
            self.feedback.style().unpolish(self.feedback)
            self.feedback.style().polish(self.feedback)
            return
        # The band is live and is not a staged analysis change.
        comparable = replace(chosen, alpha_low=self._applied.alpha_low, alpha_high=self._applied.alpha_high)
        pending = comparable.key() != self._applied.key()
        self.feedback.setObjectName("PendingSettings")
        self.feedback.setText("Settings changed. Refit to update the analysis." if pending else "")
        self.feedback.setVisible(pending)
        self.apply.setText("Working…" if self._busy else ("Refit*" if pending else "Refit"))
        self.apply.setEnabled(self._recording_available and not self._busy)
        self.feedback.style().unpolish(self.feedback)
        self.feedback.style().polish(self.feedback)

    def settings(self) -> FitSettings:
        t_start, t_length = TIME_PRESETS[self.time.currentIndex()][1:]
        if self.time.currentText() == "Custom":
            t_start = float(self.t_start.value())
            t_length = float(self.t_length.value()) or None
        if self.bursts.currentIndex() < len(BURST_PRESETS):
            _label_text, amp, amp_c, period_c, mono, min_cycles = BURST_PRESETS[self.bursts.currentIndex()]
        else:
            amp, amp_c, period_c, mono = (box.value() for box in self._custom_burst)
            min_cycles = self.burst_cycles.value()
        return FitSettings(
            t_start=float(t_start),
            t_length=None if t_length is None else float(t_length),
            fmin=float(self.fmin.value()),
            fmax=float(self.fmax.value()),
            segment_s=float(self.segment.currentData()),
            aperiodic_mode=str(self.aperiodic.currentData()),
            peak_width_low=float(self.width_low.value()),
            peak_width_high=float(self.width_high.value()),
            max_n_peaks=int(self.max_peaks.value()),
            min_peak_height=float(self.min_height.value()),
            peak_threshold=float(self.peak_threshold.value()),
            alpha_low=float(self.alpha_low.value()),
            alpha_high=float(self.alpha_high.value()),
            amp_fraction=float(amp),
            amp_consistency=float(amp_c),
            period_consistency=float(period_c),
            monotonicity=float(mono),
            min_n_cycles=int(min_cycles),
            amplitude_limit_uv=float(self.amplitude.value()),
            reference=str(self.reference.currentData()),
            epoch_s=None if self.epochs.currentData() is None else float(self.epochs.currentData()),
        )

    def current_band(self) -> tuple[float, float]:
        return float(self.alpha_low.value()), float(self.alpha_high.value())

    def set_settings(self, settings: FitSettings) -> None:
        """Restore a profile or staged project without submitting an analysis."""
        settings.validate()
        self._filling = True
        self.time.setCurrentIndex(0 if settings.t_length is None and settings.t_start == 0 else self.time.findText("Custom"))
        self.t_start.setEnabled(self.time.currentText() == "Custom")
        self.t_length.setEnabled(self.time.currentText() == "Custom")
        for field, widget in (("t_start", self.t_start), ("t_length", self.t_length), ("fmin", self.fmin), ("fmax", self.fmax),
            ("peak_width_low", self.width_low), ("peak_width_high", self.width_high), ("max_n_peaks", self.max_peaks),
            ("min_peak_height", self.min_height), ("peak_threshold", self.peak_threshold), ("alpha_low", self.alpha_low),
            ("alpha_high", self.alpha_high), ("amplitude_limit_uv", self.amplitude)):
            value = getattr(settings, field)
            if value is None:
                if field == "t_length":
                    widget.setSpecialValueText("to end")
                    widget.setValue(0)
                continue
            widget.setMaximum(max(widget.maximum(), value))
            if isinstance(widget, QDoubleSpinBox):
                widget.setDecimals(4)
            widget.setValue(value)
        for combo, value in ((self.segment, settings.segment_s), (self.aperiodic, settings.aperiodic_mode),
                             (self.reference, settings.reference), (self.epochs, settings.epoch_s)):
            index = combo.findData(value)
            if index < 0:
                combo.addItem(f"{value:g} s" if isinstance(value, (int, float)) else str(value), value)
                index = combo.count() - 1
            combo.setCurrentIndex(index)
        self.range.setCurrentIndex(self.range.findText("Custom"))
        self._show_band(settings.alpha_low, settings.alpha_high, True)
        self.bursts.setCurrentIndex(len(BURST_PRESETS))
        for box, value in zip(self._custom_burst, (settings.amp_fraction, settings.amp_consistency, settings.period_consistency, settings.monotonicity)):
            box.setValue(value)
        self.burst_cycles.setValue(settings.min_n_cycles)
        for index, preset in enumerate(BURST_PRESETS):
            if tuple(preset[1:]) == (settings.amp_fraction, settings.amp_consistency, settings.period_consistency, settings.monotonicity, settings.min_n_cycles):
                self.bursts.setCurrentIndex(index)
        self.burst_options.setVisible(self.bursts.currentText() == "Custom")
        self._filling = False
        self._refresh_feedback()

    def set_band(self, low: float, high: float, *, named: bool) -> None:
        """Show a band chosen on the Cycles page. This does not emit band_changed."""
        self._filling = True
        self._show_band(float(low), float(high), named)
        self.alpha_low.setValue(float(low))
        self.alpha_high.setValue(float(high))
        self._filling = False
        self._refresh_feedback()

    def set_custom_window(self, start: float, length: float) -> None:
        """Show a stretch on the fit bar. The fit changes when Refit is pressed."""
        self._filling = True
        custom = self.time.findText("Custom")
        if custom >= 0:
            self.time.setCurrentIndex(custom)
        self.t_start.setEnabled(True)
        self.t_length.setEnabled(True)
        self.t_length.setSpecialValueText("")
        self.t_start.setValue(max(0.0, float(start)))
        self.t_length.setValue(max(self.t_length.singleStep(), float(length)))
        self._filling = False
        self.apply.setText("Refit*")
        self._refresh_feedback()

    def _emit(self) -> None:
        chosen = self.settings()
        try:
            chosen.validate()
        except ValueError:
            self._refresh_feedback()
            return
        self.apply_requested.emit(chosen)

    def _mark_pending(self, *_args) -> None:
        if self._filling:
            return
        self._refresh_feedback()

    def _on_band(self, _index: int = 0) -> None:
        if self._filling:
            return
        data = self.band.currentData()
        if data is not None:
            self._filling = True
            self.alpha_low.setValue(float(data[0]))
            self.alpha_high.setValue(float(data[1]))
            self._filling = False
        low, high = self.current_band()
        self._refresh_feedback()
        if high > low:
            self.band_changed.emit(low, high)

    def _on_band_edge(self, _value: float = 0.0) -> None:
        if self._filling:
            return
        self._filling = True
        custom = self.band.findData(None)
        if custom >= 0 and self.band.currentIndex() != custom:
            self.band.setCurrentIndex(custom)
        self._filling = False
        low, high = self.current_band()
        self._refresh_feedback()
        if high > low:
            self.band_changed.emit(low, high)

    def _show_band(self, low: float, high: float, named: bool) -> None:
        if named:
            for index in range(self.band.count()):
                data = self.band.itemData(index)
                if data is not None and abs(float(data[0]) - low) < .00005 and abs(float(data[1]) - high) < .00005:
                    self.band.setCurrentIndex(index)
                    return
        custom = self.band.findData(None)
        if custom >= 0:
            self.band.setCurrentIndex(custom)

    def _on_time(self, _index: int = 0) -> None:
        custom = self.time.currentText() == "Custom"
        self.t_start.setEnabled(custom)
        self.t_length.setEnabled(custom)
        if custom:
            self.t_length.setSpecialValueText("to end")
            self._refresh_feedback()
            return
        if self._filling:
            return
        _label_text, start, length = TIME_PRESETS[self.time.currentIndex()]
        self._filling = True
        self.t_start.setValue(start)
        if length is None:
            self.t_length.setSpecialValueText("all")
            self.t_length.setValue(self.t_length.minimum())
        else:
            self.t_length.setSpecialValueText("")
            self.t_length.setValue(length)
        self._filling = False
        self._refresh_feedback()

    def _on_range(self, _index: int = 0) -> None:
        preset = RANGE_PRESETS[self.range.currentIndex()]
        if preset[1] is None or self._filling:
            return
        _name, fmin, fmax, segment, mode, low, high, max_peaks, height = preset
        self._filling = True
        self.fmin.setValue(fmin)
        self.fmax.setValue(fmax)
        self.width_low.setValue(low)
        self.width_high.setValue(high)
        self.max_peaks.setValue(max_peaks)
        self.min_height.setValue(height)
        self.peak_threshold.setValue(2.0)
        self.aperiodic.setCurrentIndex(0 if mode == "fixed" else 1)
        for index, (_label_text, seconds) in enumerate(SEGMENT_PRESETS):
            if seconds == segment:
                self.segment.setCurrentIndex(index)
        self._filling = False
        self._refresh_feedback()

    def _mark_custom_time(self, _value: float = 0.0) -> None:
        if self._filling:
            return
        custom = self.time.findText("Custom")
        if custom >= 0 and self.time.currentIndex() != custom:
            self.time.blockSignals(True)
            self.time.setCurrentIndex(custom)
            self.time.blockSignals(False)
            self.t_start.setEnabled(True)
            self.t_length.setEnabled(True)
        self._refresh_feedback()

    def _mark_custom_range(self, _value: float = 0.0) -> None:
        if self._filling:
            return
        custom = self.range.findText("Custom")
        if custom >= 0 and self.range.currentIndex() != custom:
            self.range.blockSignals(True)
            self.range.setCurrentIndex(custom)
            self.range.blockSignals(False)


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("H2")
    return label


def _seconds(value: float) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox()
    box.setRange(0.0, 100000.0)
    box.setDecimals(1)
    box.setSingleStep(1.0)
    box.setSuffix(" s")
    box.setValue(value)
    box.setFixedWidth(92)
    return box


def _hertz(value: float) -> QDoubleSpinBox:
    box = CompactDoubleSpinBox()
    box.setRange(0.1, 120.0)
    box.setDecimals(4)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setFixedWidth(82)
    return box
