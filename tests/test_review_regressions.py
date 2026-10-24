"""Regression coverage for the bugs found in the 2026-10-23 code walk."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from bifooof.analysis.compare import band_power
from bifooof.analysis.pac import phase_amplitude
from bifooof.analysis.prepare import prepare_recording
from bifooof.analysis.quality import QualitySpan, mask_spans
from bifooof.analysis.settings import FitSettings
from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit, _combine_channel, fit_spectra
from bifooof.analysis.timefreq import SlidingFit, channel_spectrogram
from bifooof.batch import compare_files, run_file
from bifooof.export import fit_metadata
from bifooof.io.load import _recording_from_raw, load_recording
from bifooof.io.marks import split_marked_conditions
from bifooof.io.recording import Mark, Recording


def _tone(seconds=12.0, sfreq=256.0, source="synthetic.json"):
    times = np.arange(round(seconds * sfreq)) / sfreq
    return Recording(
        data=(20.0 * np.sin(2 * np.pi * 10.0 * times))[None, :],
        ch_names=["O1"], sfreq=sfreq,
        source_path=source, source_format="synthetic",
    )


def _spectrum(recording, power=1.0):
    freqs = np.arange(1.0, 40.25, 0.25)
    item = ChannelFit(
        name="O1", index=0, usable=True, freqs=freqs,
        spectrum=np.full(freqs.shape, np.log10(power)),
        exponent=1.0, offset=0.0, r_squared=0.99,
    )
    return SpectrumFit(
        channels=[item], fmin=1.0, fmax=40.0,
        t_start=0.0, t_end=recording.duration_s,
    )


class LoaderReviewRegressions(unittest.TestCase):
    def test_cropped_dated_mne_annotations_keep_channel_scope(self):
        import mne

        date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = mne.io.RawArray(
            np.ones((2, 1200)) * 20e-6,
            mne.create_info(["O1", "O2"], 100.0, ch_types="eeg"),
            first_samp=1000, verbose="ERROR",
        )
        raw.set_meas_date(date)
        raw.set_annotations(mne.Annotations(
            [11.0, 13.0], [1.0, 0.5], ["BAD movement", "EO"],
            orig_time=date, ch_names=[("O1",), ()],
        ))
        raw.crop(tmin=0.5)
        recording = _recording_from_raw(raw, Path("cropped.fif"), "fif")
        self.assertFalse(recording.mask(0)[50:150].any())
        self.assertTrue(recording.mask(1).all())
        self.assertEqual(recording.marks, [Mark(2.5, "EO", 0.5)])

    def test_csv_rounding_is_accepted_but_nonfinite_time_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rounded.csv"
            times = np.arange(32) / 256.0
            path.write_text("time_s,O1\n" + "".join(f"{moment:.6f},{index}\n" for index, moment in enumerate(times)), encoding="utf-8")
            recording = load_recording(path)
            self.assertAlmostEqual(recording.sfreq, 256.0, delta=0.03)
            for value in ("nan", "inf"):
                path.write_text(f"time_s,O1\n0,1\n{value},2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "finite"):
                    load_recording(path)

    def test_r01_bad_annotations_use_the_recording_local_clock(self):
        import mne

        raw = mne.io.RawArray(
            np.ones((1, 1000)) * 20e-6,
            mne.create_info(["O1"], 100.0, ch_types="eeg"),
            first_samp=1000, verbose="ERROR",
        )
        raw.set_annotations(mne.Annotations([1.0], [1.0], ["BAD movement"]))
        recording = _recording_from_raw(raw, Path("offset.fif"), "fif")
        self.assertFalse(recording.mask(0)[100:200].any())
        self.assertTrue(recording.mask(0)[:100].all())

    def test_r02_mne_eye_marks_survive_loading(self):
        import mne

        raw = mne.io.RawArray(
            np.ones((1, 1000)) * 20e-6,
            mne.create_info(["O1"], 100.0, ch_types="eeg"), verbose="ERROR",
        )
        raw.set_annotations(mne.Annotations([0.0, 4.0], [0.0, 0.0], ["EC", "EO"]))
        recording = _recording_from_raw(raw, Path("marked.edf"), "edf")
        self.assertEqual([(mark.onset_s, mark.label) for mark in recording.marks], [(0.0, "EC"), (4.0, "EO")])

    def test_r03_wineeg_artifact_duration_is_excluded(self):
        n_samples = 1000
        marks_offset = 0xC0 + n_samples * 2
        raw = bytearray(marks_offset + 0x20)
        for offset, value in ((0x00, 0x0F0E), (0x02, 1), (0x04, 100), (0x0A, 2), (0x0E, 1)):
            struct.pack_into("<H", raw, offset, value)
        for offset, value in ((0x10, n_samples), (0x18, 0x80), (0x1C, 0xC0), (0x20, marks_offset), (0x24, marks_offset)):
            struct.pack_into("<I", raw, offset, value)
        raw[0x80:0x82] = b"O1"
        struct.pack_into("<f", raw, 0x98, 1.0)
        raw[0xC0:marks_offset] = np.full(n_samples, 20, dtype="<i2").tobytes()
        # An extended kind-0 mark is a duration-bearing Artifact in wineeg.py.
        struct.pack_into("<HII", raw, marks_offset, 0, 2000, 1000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.eeg"
            path.write_bytes(raw)
            recording = load_recording(path)
        self.assertEqual(recording.marks[0].label, "Artifact")
        self.assertFalse(recording.mask(0)[200:300].any())

    def test_r04_clock_notes_do_not_end_an_eye_state(self):
        recording = _tone(12.0)
        recording.marks = [Mark(0.0, "EC"), Mark(2.0, "10:36:38"), Mark(8.0, "EO")]
        sides = split_marked_conditions(recording)
        self.assertEqual(sides["ec"].duration_s, 8.0)

    def test_r05_csv_does_not_silently_compress_a_timestamp_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gap.csv"
            path.write_text("time_s,O1\n0.00,1\n0.01,2\n0.02,3\n10.03,4\n10.04,5\n", encoding="utf-8")
            # Recording is uniformly sampled. Until resampling/masking exists,
            # a timestamp discontinuity must be rejected explicitly.
            with self.assertRaises(ValueError):
                load_recording(path)


class AnalysisReviewRegressions(unittest.TestCase):
    def test_nested_crops_and_masks_keep_the_source_offset(self):
        original = _tone(20.0)
        cropped = original.cropped(5 * 256, 15 * 256).cropped(2 * 256, 8 * 256)
        masked = mask_spans(cropped, [QualitySpan("O1", 1, 2, "user", "confirmed", "")])
        prepared = prepare_recording(masked, FitSettings(t_start=1, t_length=4))
        self.assertEqual(cropped.source_start_s, 7.0)
        self.assertEqual(prepared.source_start_s, 8.0)
        self.assertFalse(prepared.mask(0)[:256].any())
        self.assertTrue(original.mask(0).all())

    def test_real_fit_metadata_counts_only_contributing_welch_runs(self):
        recording = _tone(12.0)
        recording.data[0, 4 * 256] = 1000
        settings = FitSettings(t_start=2, t_length=4)
        fit = fit_spectra(recording, settings)
        # The 2 s run before the spike contributes. The 511-sample run after
        # it is shorter than the resulting 512-sample Welch segment.
        payload = fit_metadata(recording, settings, fit)
        self.assertEqual(payload["accepted_duration_s"], 2.0)
        self.assertEqual(payload["channels"]["O1"]["accepted_spans_source_s"], [[2.0, 4.0]])
        settings.amplitude_limit_uv = 0
        self.assertEqual(fit_metadata(recording, settings, fit)["amplitude_limit_uv"], 500)

    def test_epoch_metadata_omits_the_short_tail_and_keeps_source_time(self):
        recording = _tone(24.0).cropped(4 * 256, 24 * 256)
        settings = FitSettings(epoch_s=8)
        fit = fit_spectra(recording, settings)
        payload = fit_metadata(recording, settings, fit)
        self.assertEqual(payload["accepted_duration_s"], 16.0)
        self.assertEqual(payload["source_start_s"], 4.0)
        self.assertEqual(payload["contributing_epochs"], 2)
        self.assertEqual(payload["channels"]["O1"]["excluded_spans_source_s"], [[20.0, 24.0]])

    def test_metadata_names_an_unavailable_reference_fallback(self):
        recording = _tone()
        settings = FitSettings(reference="linked")
        fit = fit_spectra(recording, settings)
        metadata = fit_metadata(recording, settings, fit)
        self.assertEqual(metadata["reference_requested"], "linked")
        self.assertEqual(metadata["reference_applied"], "stored")
        self.assertEqual(metadata["reference"], "stored")

    def test_cli_matching_also_applies_to_cycles_and_subsecond_differences(self):
        import bifooof.batch as batch

        closed, opened = _tone(8.5, source="ec.json"), _tone(8.0, source="eo.json")
        lengths = []
        original = batch.fit_cycles

        def measured(*args):
            fit = original(*args)
            lengths.append(fit.filtered.size / fit.sfreq)
            return fit

        with tempfile.TemporaryDirectory() as tmp, patch("bifooof.batch.load_recording", side_effect=[closed, opened]), patch("bifooof.batch.fit_cycles", side_effect=measured):
            compare_files("ec.json", "eo.json", tmp, plots=False)
        self.assertEqual(lengths, [8.0, 8.0])

    def test_inactive_condition_exclusions_project_and_restore(self):
        from bifooof.ui.time_panel import _reactivity

        continuous = _tone(16)
        continuous.marks = [Mark(0, "EC"), Mark(8, "EO")]
        sides = split_marked_conditions(continuous)
        masked = mask_spans(sides["eo"], [QualitySpan("O1", 2, 4, "user", "confirmed", "")])
        options = {"band": (8, 12), "continuous": continuous, "pre_s": 2, "post_s": 4, "pair": {"ec": sides["ec"], "eo": masked}}
        result = _reactivity(sides["ec"], FitSettings(), 0, options)
        column = int(np.argmin(np.abs(result["grid"] - 3)))
        self.assertTrue(np.isfinite(result["curves"][0, column]))
        self.assertFalse(np.isfinite(result["curves"][1, column]))
        options["pair"]["eo"] = sides["eo"]
        restored = _reactivity(sides["ec"], FitSettings(), 0, options)
        self.assertTrue(np.isfinite(restored["curves"][1, column]))
        self.assertTrue(continuous.mask(0).all())

    def test_r06_epoch_band_power_is_the_mean_of_linear_power(self):
        recording = _tone()
        first = _spectrum(recording, power=1.0).channels[0]
        second = _spectrum(recording, power=100.0).channels[0]
        combined = _combine_channel([first, second], FitSettings(epoch_s=8.0))
        # Mean Welch power is 50.5. Averaging log spectra yields 10 instead.
        self.assertAlmostEqual(band_power(combined, (8.0, 12.0)), 50.5)

    def test_r07_pac_map_uses_the_selected_amplitude_channel(self):
        sfreq = 256.0
        times = np.arange(int(65 * sfreq)) / sfreq
        rng = np.random.default_rng(37)
        slow = 2 * np.pi * 6 * times
        phase_signal = 10 * np.sin(slow) + rng.normal(0, 0.2, times.size)
        amp_signal = (12 + 9 * np.cos(slow)) * np.sin(2 * np.pi * 30 * times)
        recording = Recording(
            data=np.vstack([phase_signal, amp_signal]), ch_names=["O1", "O2"],
            sfreq=sfreq, source_path="pac.json", source_format="synthetic",
        )
        grid = (np.array([6.0]), np.array([30.0]))
        arguments = dict(n_surrogates=0, comodulogram_grid=grid)
        same = phase_amplitude(recording, FitSettings(), 0, (5, 7), (24, 36), amp_index=0, **arguments)
        cross = phase_amplitude(recording, FitSettings(), 0, (5, 7), (24, 36), amp_index=1, **arguments)
        self.assertGreater(cross.mi, same.mi + 0.01)
        # The map's 30 Hz cell uses 25.5–34.5 Hz, narrower than the selected
        # 24–36 Hz band. Compare the two public paths at identical band edges.
        cell = phase_amplitude(recording, FitSettings(), 0, (5, 7), (25.5, 34.5), amp_index=1, n_surrogates=0)
        self.assertAlmostEqual(cross.comodulogram[0, 0], cell.mi)
        self.assertGreater(cross.comodulogram[0, 0], same.comodulogram[0, 0] + 1e-4)

    def test_r08_metadata_counts_samples_actually_used_by_the_fit(self):
        recording = _tone(12.0)
        recording.data[0, 3 * 256] = 1000.0
        settings = FitSettings(t_start=2.0, t_length=4.0)
        prepared = prepare_recording(recording, settings)
        fit = _spectrum(recording)
        fit.t_start, fit.t_end = 2.0, 6.0
        payload = fit_metadata(recording, settings, fit)
        accepted = float(np.sum(prepared.mask(0) & np.isfinite(prepared.data[0]))) / prepared.sfreq
        self.assertAlmostEqual(payload["accepted_duration_s"], accepted)

    def test_r09_cli_comparison_matches_the_shorter_whole_recording(self):
        closed = _tone(20.0, source="ec.json")
        opened = _tone(8.0, source="eo.json")
        with tempfile.TemporaryDirectory() as tmp, patch("bifooof.batch.load_recording", side_effect=[closed, opened]):
            compare_files("ec.json", "eo.json", tmp, cycles=False, plots=False)
            metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["ec"]["time_window_end_s"], metadata["eo"]["time_window_end_s"])

    def test_r10_reexport_does_not_leave_a_previous_peak_table(self):
        recording = _tone()
        fit = _spectrum(recording)
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "peaks.csv"
            stale.write_text("previous recording's peaks", encoding="utf-8")
            with patch("bifooof.batch.load_recording", return_value=recording), patch("bifooof.batch.fit_spectra", return_value=fit):
                run_file("new.json", tmp, cycles=False, plots=False)
            self.assertFalse(stale.exists(), "A new fit with no peaks left the previous peak table behind.")

    def test_r14_event_reactivity_keeps_session_exclusions(self):
        from bifooof.ui.time_panel import _reactivity

        continuous = _tone(12.0)
        continuous.marks = [Mark(0.0, "EC"), Mark(8.0, "EO")]
        closed = split_marked_conditions(continuous)["ec"]
        masked = mask_spans(closed, [QualitySpan("O1", 2.0, 4.0, "user", "confirmed", "Review exclusion")])
        result = _reactivity(masked, FitSettings(), 0, {
            "band": (8.0, 12.0), "continuous": continuous,
            "pre_s": 2.0, "post_s": 4.0,
        })
        # At the first mark (0 s), relative 3 s is absolute 3 s. Its entire
        # two-second power block is inside the session exclusion [2, 4).
        column = int(np.argmin(np.abs(result["grid"] - 3.0)))
        self.assertFalse(np.isfinite(result["curves"][0, column]), "The original unmasked recording re-entered the event calculation.")


class UiReviewRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_failed_time_job_does_not_export_the_previous_view(self):
        from bifooof.ui.time_panel import TimePanel

        panel = TimePanel()
        self.addCleanup(panel.close)
        recording = _tone()
        with patch("bifooof.ui.time_panel.launch"), patch.object(panel, "_draw"):
            panel.set_recording(recording)
            panel._done(panel._generation, channel_spectrogram(recording, FitSettings(), 0))
            self.assertTrue(panel.rows())
            panel.spec_window.setCurrentIndex((panel.spec_window.currentIndex() + 1) % panel.spec_window.count())
            self.assertEqual(panel.rows(), [])
            panel._failed(panel._generation, "deliberate failure")
            self.assertEqual(panel.rows(), [])

    def test_pending_time_job_keeps_its_settings_snapshot(self):
        from bifooof.ui.time_panel import TimePanel

        panel = TimePanel()
        self.addCleanup(panel.close)
        with patch("bifooof.ui.time_panel.launch"), patch.object(panel, "_draw"):
            panel.set_recording(_tone())
            settings = FitSettings(reference="average")
            panel.apply_settings(settings)
            submitted = panel._worker._args[2]
            settings.reference = "linked"
            panel.set_band(13, 30)
            self.assertEqual(submitted.reference, "average")
            self.assertEqual(submitted.alpha_band(), (8, 12))
            self.assertEqual(panel.settings.alpha_band(), (13, 30))

    def test_gui_session_reexport_removes_obsolete_app_files_only(self):
        from bifooof.ui.main_window import MainWindow

        window = MainWindow()
        self.addCleanup(window.close)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "ec_peaks.csv").write_text("old peaks", encoding="utf-8")
            (folder / "notes.txt").write_text("keep this", encoding="utf-8")
            window.save_session(folder)
            self.assertFalse((folder / "ec_peaks.csv").exists())
            self.assertEqual((folder / "notes.txt").read_text(encoding="utf-8"), "keep this")
            manifest = json.loads((folder / "bifooof-manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertIn("findings.txt", manifest["files"])

    def test_r11_cached_view_change_invalidates_an_inflight_time_job(self):
        from bifooof.ui.time_panel import TimePanel

        panel = TimePanel()
        self.addCleanup(panel.close)
        recording = _tone()
        spec = channel_spectrogram(recording, FitSettings(), 0)
        with patch("bifooof.ui.time_panel.launch"), patch.object(panel, "_draw"):
            panel.set_recording(recording)
            panel._done(panel._generation, spec)
            panel.mode.setCurrentIndex(panel.mode.findData("trajectory"))
            pending_generation = panel._generation
            panel.mode.setCurrentIndex(panel.mode.findData("spectrogram"))
            self.assertTrue(panel._result is spec)
            # The previous trajectory job completes after the cache hit.
            panel._done(pending_generation, SlidingFit([], 30.0, 0.5, []))
            self.assertTrue(panel._result is spec, "The stale trajectory replaced the cached spectrogram.")

    def test_r12_time_band_follows_the_live_fit_bar_band(self):
        from bifooof.ui.main_window import MainWindow

        window = MainWindow()
        self.addCleanup(window.close)
        beta = next(index for index in range(window.fit_bar.band.count()) if window.fit_bar.band.itemText(index).startswith("Beta"))
        window.fit_bar.band.setCurrentIndex(beta)
        self.assertEqual(window.fit_bar.current_band(), (13.0, 30.0))
        self.assertEqual(window.time.settings.alpha_band(), (13.0, 30.0))

    def test_r13_failed_replacement_spectrum_is_not_still_available(self):
        from bifooof.ui.spectrum_panel import SpectrumPanel

        panel = SpectrumPanel()
        self.addCleanup(panel.close)
        recording = _tone()
        panel.recordings = {"ec": recording}
        panel.fits["ec"] = _spectrum(recording)
        with patch("bifooof.ui.spectrum_panel.launch"):
            panel._start("ec")
            panel._failed("ec", panel._generations["ec"], "replacement fit failed")
        self.assertFalse("ec" in panel.fits, "The failed replacement left its previous spectrum available.")


class ExportTransactionTests(unittest.TestCase):
    def test_failed_analysis_preserves_the_previous_bundle(self):
        from bifooof.session_export import save_bundle

        def broken(folder):
            (folder / "metadata.json").write_text("incomplete", encoding="utf-8")
            raise ValueError("analysis failed")

        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp) / "metadata.json"
            previous.write_text("previous complete result", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analysis failed"):
                save_bundle(tmp, broken)
            self.assertEqual(previous.read_text(encoding="utf-8"), "previous complete result")

    def test_publish_error_rolls_back_the_previous_bundle(self):
        import bifooof.session_export as exporting

        real_replace = exporting.os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if Path(source).parent.name == "new" and Path(source).name == "metadata.json" and not failed:
                failed = True
                raise PermissionError("file is busy")
            return real_replace(source, destination)

        def write(folder):
            (folder / "metadata.json").write_text("new", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("metadata.json", "peaks.csv", "notes.txt"):
                (folder / name).write_text("previous " + name, encoding="utf-8")
            with patch("bifooof.session_export.os.replace", side_effect=fail_once), self.assertRaises(PermissionError):
                exporting.save_bundle(folder, write)
            for name in ("metadata.json", "peaks.csv", "notes.txt"):
                self.assertEqual((folder / name).read_text(encoding="utf-8"), "previous " + name)
            self.assertFalse((folder / "bifooof-manifest.json").exists())

    def test_reexport_manifest_matches_files_and_preserves_user_notes(self):
        recording = _tone()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("peaks.csv", "cycles.csv", "cycles.png", "notes.txt"):
                (folder / name).write_text("old", encoding="utf-8")
            with patch("bifooof.batch.load_recording", return_value=recording), patch("bifooof.batch.fit_spectra", return_value=_spectrum(recording)):
                paths = run_file("new.json", folder, cycles=False, plots=False)
            self.assertEqual({path.name for path in paths}, {"spectrum.csv", "metadata.json", "bifooof-manifest.json"})
            manifest = json.loads((folder / "bifooof-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["files"]), {"spectrum.csv", "metadata.json"})
            self.assertEqual((folder / "notes.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((folder / "cycles.csv").exists())


class ReviewScreenshotFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_empty_burst_row_keeps_the_fit_clock(self):
        from matplotlib.figure import Figure

        from bifooof.analysis.cycles import CycleFit
        from bifooof.ui.time_draw import draw_raster

        fit = CycleFit(
            channel="O1", channel_index=0, band=(8.0, 12.0), sfreq=250.0,
            filtered=np.zeros(60 * 250), n_cycles=10, n_burst_cycles=0, n_burst_episodes=0,
            burst_duration_s=0.0, burst_fraction=0.0, period_ms=100.0, rise_decay=0.5,
            peak_trough=0.5, volt_amp=20.0, burst_period_ms=None, burst_rise_decay=None,
            burst_peak_trough=None, burst_volt_amp=None, t_offset=120.0,
        )
        fig = Figure()
        draw_raster(fig, [fit], None)
        left, right = fig.axes[0].get_xlim()
        self.assertAlmostEqual(left, 120.0, delta=0.1)
        self.assertAlmostEqual(right, 180.0, delta=0.1)
        self.assertEqual(fig.axes[0].get_yticklabels()[0].get_text(), "O1")

    def test_burst_checklist_follows_a_single_channel_change(self):
        from PySide6.QtCore import Qt

        from bifooof.ui.time_panel import TimePanel

        panel = TimePanel()
        panel.set_recording(Recording(np.zeros((3, 256)), ["Fp1", "F3", "Cz"], 256.0, "x.csv", "csv"))
        panel.set_selected(2)
        self.app.processEvents()
        self.assertEqual(panel.picks.item(2).checkState(), Qt.Checked)
        self.assertEqual(panel.picks.item(0).checkState(), Qt.Unchecked)
        panel.picks.item(0).setCheckState(Qt.Checked)
        panel.set_selected(1)
        self.app.processEvents()
        self.assertEqual(panel.picks.item(0).checkState(), Qt.Checked)
        self.assertEqual(panel.picks.item(2).checkState(), Qt.Checked)

    def test_burst_checklist_scrolls_to_a_single_channel_change(self):
        from PySide6.QtCore import Qt

        from bifooof.ui.time_panel import TimePanel

        names = [f"Ch{index:02d}" for index in range(20)]
        panel = TimePanel()
        panel.resize(480, 320)
        panel.picks.setFixedHeight(48)
        panel.show()
        panel.set_recording(Recording(np.zeros((20, 256)), names, 256.0, "x.csv", "csv"))
        panel.set_selected(19)
        self.app.processEvents()
        item = panel.picks.item(19)
        self.assertEqual(item.checkState(), Qt.Checked)
        self.assertTrue(panel.picks.viewport().rect().intersects(panel.picks.visualItemRect(item)))
        panel.hide()

    def test_command_line_and_window_report_the_same_peak(self):
        from bifooof.jobs import JobToken, current_job

        sfreq = 128.0
        samples = int(sfreq * 16)
        times = np.arange(samples) / sfreq
        alpha = 40.0 * np.sin(2 * np.pi * 10.0 * times)
        rng = np.random.default_rng(1)
        data = np.vstack([
            alpha + rng.normal(scale=8.0, size=samples),
            0.4 * alpha + rng.normal(scale=8.0, size=samples),
        ])
        recording = Recording(data, ["O1", "O2"], sfreq, "peaks.csv", "csv")
        settings = FitSettings(t_length=16.0)
        direct = fit_spectra(recording, settings)
        token = current_job.set(JobToken())
        try:
            window = fit_spectra(recording, settings)
        finally:
            current_job.reset(token)
        self.assertTrue(direct.channels[0].peaks)
        for left, right in zip(direct.channels, window.channels):
            self.assertEqual(len(left.peaks), len(right.peaks))
            for found, shown in zip(left.peaks, right.peaks):
                self.assertAlmostEqual(found[0], shown[0], delta=0.05)
                self.assertAlmostEqual(found[1], shown[1], delta=0.05)
                self.assertAlmostEqual(found[2], shown[2], delta=0.05)

    def test_one_fit_window_is_not_drawn_as_variability(self):
        from bifooof.ui.study_panel import StudyPanel

        one = SpectrumFit(
            channels=[ChannelFit("O1", 0, True, exponent=0.27, r_squared=0.9)],
            fmin=1, fmax=40, t_start=120, t_end=180, settings=FitSettings(),
        )
        panel = StudyPanel(_workspace(one))
        panel.refresh()
        self.assertIn("not a variability series", panel.caption.text())
        self.assertFalse(panel.canvas.fig.axes[0].lines)
        self.assertGreater(panel.table.rowCount(), 0)

        two = SpectrumFit(
            channels=[], fmin=1, fmax=40, settings=FitSettings(),
            epochs=[
                SpectrumFit(channels=[ChannelFit("O1", 0, True, exponent=0.2, r_squared=0.9)], fmin=1, fmax=40, t_start=120, t_end=150),
                SpectrumFit(channels=[ChannelFit("O1", 0, True, exponent=0.4, r_squared=0.9)], fmin=1, fmax=40, t_start=150, t_end=180),
            ],
        )
        panel.workspace.window.spectrum.fits["ec"] = two
        panel.refresh()
        self.assertIn("not confidence intervals", panel.caption.text())
        self.assertTrue(panel.canvas.fig.axes[0].lines)


class _Spectrum:
    def __init__(self, fit):
        self.fits = {"ec": fit}
        self.settings = FitSettings()


class _Window:
    active = "ec"
    recording = _tone()

    def __init__(self, fit):
        self.spectrum = _Spectrum(fit)

    def _channel_name(self):
        return "O1"


class _State:
    value = {"regions": {}}


class _Workspace:
    def __init__(self, fit):
        self.window = _Window(fit)
        self.state = _State()


def _workspace(fit):
    return _Workspace(fit)


if __name__ == "__main__":
    unittest.main()
