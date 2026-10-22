"""Controls for the time window, spectrum range, and burst rule."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout,
)

from ..analysis.settings import BURST_PRESETS, RANGE_PRESETS, SEGMENT_PRESETS, TIME_PRESETS, FitSettings


class FitBar(QFrame):
    """The choices resting-state specparam and bycycle work actually changes."""

    apply_requested = Signal(object)

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
            "3–30 and 2–20 Hz are for theta, alpha, and beta without gamma."
        )
        self.fmin = _hertz(1.0)
        self.fmax = _hertz(40.0)
        self.fmin.setToolTip("Lowest frequency in the specparam fit.")
        self.fmax.setToolTip("Highest frequency in the specparam fit.")

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
        self.alpha_low = _hertz(8.0)
        self.alpha_high = _hertz(12.0)
        self.alpha_low.setToolTip("Low edge of the band used for alpha reactivity, in Hz.")
        self.alpha_high.setToolTip("High edge of the band used for alpha reactivity, in Hz.")

        self.bursts = QComboBox()
        for label, *_rest in BURST_PRESETS:
            self.bursts.addItem(label)
        self.bursts.setToolTip(
            "Library keeps every consistent run of 3 cycles. "
            "Resting alpha and paper alpha require a taller amplitude before a run counts as a burst."
        )

        self.apply = QPushButton("Refit")
        self.apply.setObjectName("Secondary")
        self.apply.setToolTip("Run specparam and bycycle again with these settings.")
        self.apply.clicked.connect(self._emit)

        self.time.currentIndexChanged.connect(self._on_time)
        self.range.currentIndexChanged.connect(self._on_range)
        self.segment.currentIndexChanged.connect(self._mark_custom_range)
        self.aperiodic.currentIndexChanged.connect(self._mark_custom_range)
        for box in (self.t_start, self.t_length):
            box.valueChanged.connect(self._mark_custom_time)
        for box in (self.fmin, self.fmax, self.width_low, self.width_high, self.min_height, self.max_peaks):
            box.valueChanged.connect(self._mark_custom_range)
        self.peak_threshold.valueChanged.connect(self._mark_custom_range)
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
        bottom.addWidget(_label("ALPHA"))
        bottom.addWidget(self.alpha_low)
        bottom.addWidget(_label("–"))
        bottom.addWidget(self.alpha_high)
        bottom.addWidget(_label("Hz"))
        bottom.addSpacing(12)
        bottom.addWidget(_label("BURSTS"))
        bottom.addWidget(self.bursts)
        bottom.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 8, 16, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(bottom)

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
        )

    def _emit(self) -> None:
        self.apply_requested.emit(self.settings())

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
