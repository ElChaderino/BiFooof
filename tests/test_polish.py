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

from bifooof.analysis.cycles import BurstEpisode
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

    def test_a_failed_reference_is_described_as_the_one_applied(self):
        panel = self.window().spectrum
        panel.settings = FitSettings(reference="rest")
        panel.fit = SpectrumFit(
            [ChannelFit("O1", 0, True, exponent=1.0, r_squared=0.9)],
            1, 40,
            notes="REST needs at least 4 channels on the 10-20 layout. The stored reference was kept.",
            provenance={"reference_applied": "stored", "reference_requested": "rest"},
        )
        panel._describe()
        text = panel.caption.text()
        self.assertIn("reference as stored", text)
        self.assertIn("stored reference was kept", text)
        self.assertNotIn("template spherical model", text)
        compare = self.window().compare
        compare.settings = FitSettings(reference="rest")
        compare.fits = {
            "ec": SpectrumFit([ChannelFit("O1", 0, True)], 1, 40, provenance={"reference_applied": "stored"}),
            "eo": SpectrumFit([ChannelFit("O1", 0, True)], 1, 40, provenance={"reference_applied": "rest"}),
        }
        sentence = compare._reference_sentence()
        self.assertIn("EC: reference as stored.", sentence)
        self.assertIn("EO: REST, template spherical model.", sentence)
        compare.fits["eo"].provenance = {"reference_applied": "stored"}
        self.assertEqual(compare._reference_sentence(), "Reference as stored.")

    def test_a_burst_click_uses_the_fitted_unit(self):
        from types import SimpleNamespace

        panel = self.window().time
        panel.recording = tone()
        panel.bursts.setChecked(True)
        panel.settings = FitSettings(reference="csd")
        panel.cycle_fit = SimpleNamespace(
            channel="O1",
            unit="µV",
            episodes=[BurstEpisode(1.0, 2.0, 3, 100.0, 12.0)],
        )
        text = panel._clicked_bursts(0.5, 2.5)
        self.assertIn("12.0 µV", text)
        self.assertNotIn("µV/m²", text)


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_a_signed_head_map_is_centered_on_zero(self):
        from matplotlib.figure import Figure

        from bifooof.ui.headmap import draw_head

        fig = Figure()
        ax = fig.add_subplot(111)
        draw_head(ax, ["O1", "O2"], [-1.0, 1.0], "Change", "viridis", diverging=True, unit="EO − EC")
        clim = next(coll.get_clim() for coll in ax.collections if coll.get_array() is not None and np.size(coll.get_array()))
        self.assertAlmostEqual(clim[0], -clim[1])
        self.assertLess(clim[0], 0)
        self.assertIn("EO − EC", ax.figure.axes[-1].get_ylabel())

    def test_coherence_names_both_files_and_the_applied_reference(self):
        from matplotlib.figure import Figure

        from bifooof.analysis.coherence import CoherenceResult
        from bifooof.ui.time_draw import draw_coherence

        freqs = np.linspace(1.0, 30.0, 16)
        this = CoherenceResult("C3", "C4", freqs, np.full(16, 0.4), np.linspace(-40, 40, 16), np.zeros(16), None, 4, 8.0, "linked")
        other = CoherenceResult("C3", "C4", freqs, np.full(16, 0.2), np.linspace(-10, 10, 16), np.zeros(16), None, 4, 8.0, "stored")
        fig = Figure()
        draw_coherence(fig, this, other=other, other_label="Eyes open")
        self.assertIn("linked-ear", fig.axes[0].get_title(loc="left"))
        labels = [text.get_text() for ax in fig.axes[:2] for text in ax.get_legend().get_texts()]
        self.assertIn("This file", labels)
        self.assertIn("Eyes open", labels)
        self.assertEqual(fig.axes[1].get_ylim(), (-180.0, 180.0))
        self.assertTrue(fig.axes)

    def test_cycle_and_reactivity_axes_use_the_fitted_unit(self):
        from matplotlib.figure import Figure

        from bifooof.analysis.cycles import CycleFit
        from bifooof.plots import render_cycles
        from bifooof.ui.cycle_panel import CyclePanel
        from bifooof.ui.time_draw import draw_reactivity

        sfreq = 256.0
        filtered = np.sin(2 * np.pi * 10.0 * np.arange(sfreq) / sfreq)
        fit = CycleFit(
            channel="O1", channel_index=0, band=(8.0, 12.0), sfreq=sfreq, filtered=filtered,
            n_cycles=1, n_burst_cycles=1, n_burst_episodes=1, burst_duration_s=0.4, burst_fraction=0.4,
            period_ms=100.0, rise_decay=0.5, peak_trough=0.5, volt_amp=10.0,
            burst_period_ms=100.0, burst_rise_decay=0.5, burst_peak_trough=0.5, burst_volt_amp=10.0,
            spans=[(10, 80, True)], peaks=[40], unit="µV/m²",
        )
        fig = Figure()
        render_cycles(fig, fit)
        self.assertEqual(fig.axes[0].get_ylabel(), "µV/m²")
        self.assertIn("O1", fig.axes[0].get_title(loc="left"))
        self.assertIn("µV/m²", fig._bifooof_stamp)
        panel = CyclePanel()
        self.addCleanup(panel.close)
        panel.fit = fit
        panel._draw()
        self.assertEqual(panel.canvas.fig.axes[0].get_ylabel(), "µV/m²")
        power = Figure()
        draw_reactivity(power, {
            "kind": "single",
            "single": (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            "title": "O1",
            "unit": "µV²/(m⁴·Hz)",
        }, None)
        self.assertEqual(power.axes[0].get_ylabel(), "Band power (µV²/(m⁴·Hz))")
        self.assertTrue(power.axes)

    def test_a_saved_figure_keeps_its_line_off_the_screen(self):
        import tempfile
        from matplotlib.figure import Figure

        from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit
        from bifooof.export import save_figure
        from bifooof.plots import render_spectrum

        freqs = np.linspace(1.0, 40.0, 32)
        item = ChannelFit(
            name="O1", index=0, usable=True, exponent=1.0, r_squared=0.9,
            freqs=freqs, spectrum=np.linspace(1.5, 0.2, freqs.size),
            model=np.linspace(1.4, 0.2, freqs.size), aperiodic=np.linspace(1.2, 0.2, freqs.size),
            peaks=[(10.0, 0.3, 2.0)],
        )
        fit = SpectrumFit(
            channels=[item], fmin=1.0, fmax=40.0, t_start=0.0, t_end=8.0,
            provenance={"reference_applied": "csd"},
        )
        fig = Figure()
        render_spectrum(fig, fit, 0, "Eyes closed")
        self.assertTrue(fig.axes)
        self.assertTrue(any("µV²/(m⁴·Hz)" in ax.get_ylabel() for ax in fig.axes))
        self.assertIn("alpha", [text.get_text() for ax in fig.axes for text in ax.texts])
        self.assertIn("current source density", fig._bifooof_stamp)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_figure(fig, Path(tmp) / "spectrum.png")
            self.assertTrue(path.is_file())
        self.assertFalse(any(text.get_text() == fig._bifooof_stamp for text in fig.texts))


class MathEdgeTests(unittest.TestCase):
    def test_a_blank_reference_ear_returns_on_the_continuous_file(self):
        from bifooof.analysis.quality import project_exclusions

        ear = np.full(64, 4.0)
        source = Recording(np.ones((2, 64)), ["O1", "O2"], 128.0, "same.csv", "csv", reference_channels={"A1": ear.copy()})
        condition = Recording(np.ones((2, 64)), ["O1", "O2"], 128.0, "same.csv", "csv", reference_channels={"A1": ear.copy()})
        condition.reference_channels["A1"][10:20] = np.nan
        projected = project_exclusions(source, [condition])
        self.assertTrue(np.isnan(projected.reference_channels["A1"][10:20]).all())
        self.assertTrue(np.isfinite(projected.reference_channels["A1"][:10]).all())
        self.assertTrue(np.isfinite(source.reference_channels["A1"]).all())
        already = ear.copy()
        already[5:8] = np.nan
        untouched = Recording(np.ones((2, 64)), ["O1", "O2"], 128.0, "same.csv", "csv", reference_channels={"A1": already.copy()})
        copy = Recording(np.ones((2, 64)), ["O1", "O2"], 128.0, "same.csv", "csv", reference_channels={"A1": already.copy()})
        again = project_exclusions(untouched, [copy])
        self.assertTrue(np.array_equal(np.isnan(again.reference_channels["A1"]), np.isnan(already)))

    def test_global_scale_is_not_applied_twice_and_a_reversed_band_is_refused(self):
        from bifooof.analysis.prepare import apply_reference
        from bifooof.analysis.timefreq import block_band_power

        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        values = np.vstack([
            10.0 * np.sin(2 * np.pi * 10.0 * times),
            4.0 * np.sin(2 * np.pi * 6.0 * times),
        ])
        settings = FitSettings(reference="global", amplitude_limit_uv=0, segment_s=4.0)
        once = apply_reference(Recording(values, ["O1", "O2"], sfreq, "scale.csv", "csv"), "global", 0.0, settings)
        twice = apply_reference(once, "global", 0.0, settings)
        self.assertEqual(twice.reference_applied, "global")
        self.assertIn("already applied", twice.notes[-1])
        np.testing.assert_allclose(twice.data, once.data)
        with self.assertRaises(ValueError):
            block_band_power(tone(), FitSettings(), 0, (12.0, 8.0), 2.0, 0.5)

    def test_a_crop_write_does_not_change_the_open_recording(self):
        source = tone()
        original = source.data.copy()
        window = source.cropped(10, 40)
        window.data[:] = 0.0
        np.testing.assert_array_equal(source.data, original)

    def test_export_and_findings_leave_out_nonfinite_numbers_and_a_failed_request(self):
        import tempfile

        from bifooof.analysis.cycles import CycleFit
        from bifooof.analysis.findings import compile_report, report_text
        from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit
        from bifooof.export import write_csv, write_json
        from bifooof.report import html_table

        with tempfile.TemporaryDirectory() as tmp:
            table = write_csv(Path(tmp) / "rows.csv", [{"exponent": float("nan"), "height": float("inf"), "name": "O1"}])
            text = table.read_text(encoding="utf-8").lower()
            self.assertNotIn("nan", text)
            self.assertNotIn("inf", text)
            payload = write_json(Path(tmp) / "rows.json", {"exponent": float("nan"), "height": float("inf")})
            saved = payload.read_text(encoding="utf-8").lower()
            self.assertNotIn("nan", saved)
            self.assertNotIn("inf", saved)
        page = html_table([{"exponent": float("nan")}]).lower()
        self.assertNotIn("nan", page)

        item = ChannelFit(name="O1", index=0, usable=True, exponent=float("nan"), r_squared=0.9, peaks=[])
        fit = SpectrumFit(
            channels=[item], fmin=1.0, fmax=40.0, t_start=0.0, t_end=4.0,
            provenance={"reference_applied": "stored"},
        )
        recording = Recording(np.zeros((1, 16)), ["O1"], 256.0, "closed.csv", "csv")
        cycles = CycleFit(
            channel="O1", channel_index=0, band=(8.0, 12.0), sfreq=256.0, filtered=np.zeros(16),
            n_cycles=2, n_burst_cycles=0, n_burst_episodes=0, burst_duration_s=0.0, burst_fraction=0.0,
            period_ms=float("nan"), rise_decay=None, peak_trough=None, volt_amp=float("inf"),
            burst_period_ms=None, burst_rise_decay=None, burst_peak_trough=None, burst_volt_amp=None,
        )
        sections = compile_report(
            {"ec": recording}, {"ec": fit}, {"ec": cycles},
            (8.0, 12.0), "O1", FitSettings(reference="left"),
        )
        report = report_text(sections)
        folded = report.lower()
        self.assertNotRegex(folded, r"\bnan\b")
        self.assertNotRegex(folded, r"\binf\b")
        self.assertNotIn("was asked for", folded)
        self.assertIn("as stored", report)
        for section in sections:
            for line in section.lines:
                self.assertIsInstance(line, str)

    def test_a_second_spatial_transform_is_refused(self):
        from bifooof.analysis.prepare import apply_reference

        sfreq = 256.0
        times = np.arange(int(1.0 * sfreq)) / sfreq
        names = ["Fp1", "F3", "C3", "O1"]
        signal = np.vstack([8.0 * np.sin(2 * np.pi * 10.0 * times + phase) for phase in (0.0, 0.4, 0.8, 1.2)])
        source = Recording(signal, names, sfreq, "spatial.csv", "csv")
        for mode in ("csd", "rest"):
            once = apply_reference(source, mode)
            self.assertEqual(once.reference_applied, mode)
            twice = apply_reference(once, mode)
            self.assertEqual(twice.reference_applied, mode)
            self.assertIn("already applied", twice.notes[-1])
            np.testing.assert_allclose(twice.data, once.data)


if __name__ == "__main__":
    unittest.main()
