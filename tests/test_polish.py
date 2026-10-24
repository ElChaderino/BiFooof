"""Behavioral checks for the desktop usability pass."""

from dataclasses import fields, replace
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from bifooof.analysis.settings import FitSettings
from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit
from bifooof.analysis.timefreq import channel_spectrogram
from bifooof.io.recording import Recording
from bifooof.ui.fit_bar import FitBar
from bifooof.ui.main_window import MainWindow


def tone():
    times = np.arange(256 * 12) / 256.0
    return Recording((20 * np.sin(2 * np.pi * 10 * times))[None, :], ["O1"], 256.0, "synthetic.json", "synthetic")


class ValidationTests(unittest.TestCase):
    def test_every_numeric_setting_rejects_nonfinite_values(self):
        settings = FitSettings(t_length=10.0, epoch_s=30.0)
        for field in fields(settings):
            if not isinstance(getattr(settings, field.name), (float, int)):
                continue
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field.name, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    replace(settings, **{field.name: value}).validate()

    def test_physical_ranges_and_counts_are_checked(self):
        for change in (
            {"fmin": 0}, {"alpha_low": -1}, {"peak_width_low": 0},
            {"min_peak_height": -0.1}, {"amp_fraction": 1.01},
            {"amp_consistency": -0.1}, {"period_consistency": 1.1},
            {"monotonicity": 1.1}, {"min_n_cycles": 2.5}, {"max_n_peaks": 1.5},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(FitSettings(), **change).validate()


class DesktopPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        window = MainWindow()
        self.addCleanup(window.close)
        return window

    def test_staged_settings_can_be_applied_and_reverted(self):
        bar = FitBar()
        self.addCleanup(bar.close)
        bar.set_context(True)
        self.assertTrue(bar.feedback.isHidden())
        bar.reference.setCurrentIndex(1)
        self.assertFalse(bar.feedback.isHidden())
        bar.reference.setCurrentIndex(0)
        self.assertTrue(bar.feedback.isHidden())
        bar.set_custom_window(3.0, 5.0)
        chosen = bar.settings()
        self.assertEqual((chosen.t_start, chosen.t_length), (3.0, 5.0))
        bar.mark_applied(chosen)
        self.assertTrue(bar.feedback.isHidden())

    def test_live_band_is_not_a_pending_refit_and_invalid_band_cannot_submit(self):
        bar = FitBar()
        self.addCleanup(bar.close)
        bar.set_context(True)
        submitted = []
        bar.apply_requested.connect(submitted.append)
        bar.set_band(13, 30, named=True)
        self.assertTrue(bar.feedback.isHidden())
        bar.alpha_low.setValue(40)
        self.assertFalse(bar.apply.isEnabled())
        bar._emit()
        self.assertFalse(submitted)
        bar.alpha_high.setValue(45)
        self.assertTrue(bar.apply.isEnabled())

    def test_malformed_file_inspection_shows_an_error_and_does_not_queue(self):
        window = self.window()
        with patch("bifooof.ui.main_window.QFileDialog.getOpenFileName", return_value=("broken.csv", "")), \
             patch("bifooof.ui.main_window.needs_sfreq", side_effect=ValueError("Missing columns")), \
             patch.object(window, "_show_error") as error, patch.object(window, "_queue_load") as queue:
            window.open_recording("ec")
        error.assert_called_once()
        self.assertIn("Missing columns", error.call_args.args[1])
        queue.assert_not_called()

    def test_image_data_and_session_write_failures_are_reported(self):
        window = self.window()
        window.recording = tone()
        window.recordings = {"ec": window.recording}
        for action, failing in ((window.save_image, "save_figure"), (window.save_data, "write_csv")):
            with self.subTest(action=action.__name__), patch.object(window, "_choose_save", return_value="locked.out"), \
                 patch.object(window, "_show_error") as error, \
                 patch(f"bifooof.ui.main_window.{failing}", side_effect=PermissionError("Folder is read-only")):
                action()
                self.assertIn("Folder is read-only", error.call_args.args[1])
        with patch("bifooof.ui.main_window.QFileDialog.getExistingDirectory", return_value="locked"), \
             patch.object(window, "save_session", side_effect=PermissionError("Folder is read-only")), \
             patch.object(window, "_show_error") as error:
            window._save_session_dialog()
            self.assertIn("Folder is read-only", error.call_args.args[1])

    def test_exports_follow_the_current_tab_and_pending_time_results(self):
        window = self.window()
        window._refresh_actions()
        self.assertFalse(window.save_image_button.isEnabled())
        self.assertFalse(window.save_session_button.isEnabled())
        window.recording = tone()
        window.recordings = {"ec": window.recording}
        window._refresh_actions()
        self.assertTrue(window.save_data_button.isEnabled())
        window.tabs.setCurrentWidget(window.spectrum)
        self.assertFalse(window.save_image_button.isEnabled())
        with patch("bifooof.ui.time_panel.launch"):
            window.time.set_recording(window.recording)
        window.tabs.setCurrentWidget(window.time)
        self.assertFalse(window.save_data_button.isEnabled())
        window.time._done(window.time._generation, channel_spectrogram(window.recording, FitSettings(), 0))
        with patch.object(window.time, "rows", side_effect=AssertionError("Do not serialize on UI refresh")):
            window._refresh_actions()
        self.assertTrue(window.save_image_button.isEnabled())
        self.assertTrue(window.save_data_button.isEnabled())
        window.time._failed(window.time._generation, "deliberate failure")
        window._refresh_actions()
        self.assertFalse(window.save_data_button.isEnabled())

    def test_close_waits_without_blocking_the_event_loop(self):
        window = self.window()
        worker = Mock()
        worker.isRunning.return_value = True
        window._live_workers = [worker]
        event = QCloseEvent()
        window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        worker.wait.assert_not_called()
        self.assertTrue(window._state_timer.isActive())
        worker.isRunning.return_value = False
        window.closeEvent(event)
        self.assertTrue(event.isAccepted())

    def test_late_load_during_close_does_not_start_another_analysis(self):
        window = self.window()
        window._closing = True
        window._load_generations["ec"] = 1
        with patch.object(window, "_present") as present:
            window._loaded("ec", 1, tone())
        present.assert_not_called()

    def test_negative_or_extreme_knee_does_not_break_the_caption(self):
        panel = self.window().spectrum
        for knee, exponent in ((-1.0, 1.5), (1e300, .001), (1.0, 0.0)):
            panel.fit = SpectrumFit([ChannelFit("O1", 0, True, knee=knee, exponent=exponent)], 1, 40)
            panel._describe()
            self.assertIn("O1", panel.caption.text())


if __name__ == "__main__":
    unittest.main()
