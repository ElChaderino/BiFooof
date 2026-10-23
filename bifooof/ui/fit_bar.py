"""Controls for the time window, spectrum range, and burst rule."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout,
)

from ..analysis.cycles import BANDS
from ..analysis.settings import BURST_PRESETS, RANGE_PRESETS, SEGMENT_PRESETS, TIME_PRESETS, FitSettings
from ..analysis.specparam_fit import EPOCH_PEAKS


class FitBar(QFrame):
    """The choices resting-state specparam and bycycle work actually changes."""

    apply_requested = Signal(object)
    band_changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setObjectName("Header")
        self._filling = False

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
            self.segment.addItem(label)
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
        self.min_height = QDoubleSpinBox()
        self.min_height.setRange(0.0, 2.0)
        self.min_height.setDecimals(2)
        self.min_height.setSingleStep(0.05)
        self.min_height.setValue(0.05)
        self.min_height.setToolTip("Minimum peak height above the aperiodic fit, in log10 power.")
        self.peak_threshold = QDoubleSpinBox()
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
        self.amplitude = QDoubleSpinBox()
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

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(_label("TIME"))
        top.addWidget(self.time)
        top.addWidget(_label("Start"))
        top.addWidget(self.t_start)
        top.addWidget(_label("Length"))
        top.addWidget(self.t_length)
        top.addSpacing(8)
        top.addWidget(_label("FIT"))
        top.addWidget(self.range)
        top.addWidget(self.fmin)
        top.addWidget(_label("to"))
        top.addWidget(self.fmax)
        top.addWidget(_label("Hz"))
        top.addSpacing(8)
        top.addWidget(_label("SEGMENT"))
        top.addWidget(self.segment)
        top.addWidget(_label("SLOPE"))
        top.addWidget(self.aperiodic)
        top.addStretch(1)
        top.addWidget(self.apply)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(_label("PEAKS"))
        bottom.addWidget(_label("width"))
        bottom.addWidget(self.width_low)
        bottom.addWidget(_label("–"))
        bottom.addWidget(self.width_high)
        bottom.addWidget(_label("Hz"))
        bottom.addWidget(_label("max"))
        bottom.addWidget(self.max_peaks)
        bottom.addWidget(_label("height"))
        bottom.addWidget(self.min_height)
        bottom.addWidget(_label("threshold"))
        bottom.addWidget(self.peak_threshold)
        bottom.addSpacing(12)
        bottom.addWidget(_label("BAND"))
        bottom.addWidget(self.band)
        bottom.addWidget(self.alpha_low)
        bottom.addWidget(_label("–"))
        bottom.addWidget(self.alpha_high)
        bottom.addWidget(_label("Hz"))
        bottom.addSpacing(12)
        bottom.addWidget(_label("BURSTS"))
        bottom.addWidget(self.bursts)
        bottom.addStretch(1)

        extra = QHBoxLayout()
        extra.setSpacing(8)
        extra.addWidget(_label("REFERENCE"))
        extra.addWidget(self.reference)
        extra.addSpacing(12)
        extra.addWidget(_label("EPOCHS"))
        extra.addWidget(self.epochs)
        extra.addSpacing(12)
        extra.addWidget(_label("LIMIT"))
        extra.addWidget(self.amplitude)
        extra.addWidget(_label("µV"))
        extra.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 8, 16, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(bottom)
        layout.addLayout(extra)

    def settings(self) -> FitSettings:
        t_start, t_length = TIME_PRESETS[self.time.currentIndex()][1:]
        if self.time.currentText() == "Custom":
            t_start = float(self.t_start.value())
            t_length = float(self.t_length.value())
        _label_text, amp, amp_c, period_c, mono, min_cycles = BURST_PRESETS[self.bursts.currentIndex()]
        return FitSettings(
            t_start=float(t_start),
            t_length=None if t_length is None else float(t_length),
            fmin=float(self.fmin.value()),
            fmax=float(self.fmax.value()),
            segment_s=SEGMENT_PRESETS[self.segment.currentIndex()][1],
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

    def set_band(self, low: float, high: float, *, named: bool) -> None:
        """Show a band chosen on the Cycles page. This does not emit band_changed."""
        self._filling = True
        self._show_band(float(low), float(high), named)
        self.alpha_low.setValue(float(low))
        self.alpha_high.setValue(float(high))
        self._filling = False

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

    def _emit(self) -> None:
        self.apply.setText("Refit")
        self.apply_requested.emit(self.settings())

    def _mark_pending(self, *_args) -> None:
        if self._filling:
            return
        self.apply.setText("Refit*")

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
        self.band_changed.emit(low, high)

    def _show_band(self, low: float, high: float, named: bool) -> None:
        if named:
            for index in range(self.band.count()):
                data = self.band.itemData(index)
                if data is not None and abs(float(data[0]) - low) < 0.05 and abs(float(data[1]) - high) < 0.05:
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
            self.t_length.setSpecialValueText("")
            if self.t_length.value() <= 0:
                self._filling = True
                self.t_length.setValue(60.0)
                self._filling = False
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
    box = QDoubleSpinBox()
    box.setRange(0.0, 100000.0)
    box.setDecimals(1)
    box.setSingleStep(1.0)
    box.setSuffix(" s")
    box.setValue(value)
    box.setFixedWidth(92)
    return box


def _hertz(value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.1, 120.0)
    box.setDecimals(1)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setFixedWidth(72)
    return box
