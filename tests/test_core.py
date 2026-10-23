"""Loader, fit, and offscreen window checks."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from PySide6.QtWidgets import QApplication

from bifooof.analysis.compare import ChannelComparison, alpha_peak, compare_spectra, nearest_outside_peak
from bifooof.analysis.cycles import burst_cutoffs, burst_episodes, burst_note, fit_cycles
from bifooof.analysis.prepare import apply_reference, peak_at_width_floor, prepare_recording, voltage_scale_note
from bifooof.analysis.segments import episode_count
from bifooof.analysis.settings import FitSettings, band_label, band_phrase, matched_settings
from bifooof.analysis.specparam_fit import ChannelFit, fit_spectra
from bifooof.analysis.spectra import welch_spectra
from bifooof.export import burst_episode_rows, comparison_rows, cycle_summary_row
from bifooof.io.channels import clean_channel_name
from bifooof.io.condition import guess_condition
from bifooof.io.load import _recording_from_raw, finalize, load_recording, needs_sfreq
from bifooof.io.marks import split_marked_conditions
from bifooof.io.recording import Mark, Recording
from bifooof.io.sniff import sniff
from bifooof.ui.main_window import MainWindow


def _synthetic(seconds: float = 8.0, sfreq: float = 256.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    times = np.arange(int(seconds * sfreq)) / sfreq
    fp1 = 20.0 * np.sin(2 * np.pi * 10.0 * times)
    flat = np.zeros_like(times)
    cz = 12.0 * np.sin(2 * np.pi * 6.0 * times) + 4.0 * np.sin(2 * np.pi * 10.0 * times)
    data = np.vstack([fp1, flat, cz])
    return times, data, ["Fp1-LE", "F3", "EEG Cz"]


class FormatTests(unittest.TestCase):
    def test_channel_aliases(self):
        self.assertEqual(clean_channel_name("Fp1-LE"), "Fp1")
        self.assertEqual(clean_channel_name("EEG Fp2"), "Fp2")
        self.assertEqual(clean_channel_name("T3"), "T7")
        self.assertEqual(clean_channel_name("T6"), "P8")
        self.assertEqual(clean_channel_name("C3-Av"), "C3")
        self.assertEqual(clean_channel_name("Cz-REF"), "Cz")

    def test_eeg_magic_and_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wineeg = root / "native.eeg"
            wineeg.write_bytes(bytes([0x0E, 0x0F]) + b"\x00" * 6)
            self.assertEqual(sniff(wineeg), "wineeg")

            orphan = root / "traces.eeg"
            orphan.write_bytes(b"BRAINVIS")
            self.assertEqual(sniff(orphan), "unknown-eeg")
            with self.assertRaises(ValueError) as caught:
                load_recording(orphan)
            self.assertIn("no BrainVision .vhdr", str(caught.exception))

            header = root / "traces.vhdr"
            header.write_text("Brain Vision Data Exchange Header File Version 1.0\n", encoding="ascii")
            self.assertEqual(sniff(orphan), "brainvision")

    def test_wineeg_is_read_without_an_external_converter(self):
        raw = bytearray(0xC8)
        struct.pack_into("<H", raw, 0x00, 0x0F0E)
        struct.pack_into("<H", raw, 0x02, 1)
        struct.pack_into("<H", raw, 0x04, 256)
        struct.pack_into("<H", raw, 0x0A, 2)
        struct.pack_into("<I", raw, 0x10, 4)
        struct.pack_into("<I", raw, 0x18, 0x80)
        struct.pack_into("<I", raw, 0x1C, 0xC0)
        struct.pack_into("<I", raw, 0x20, 0xC8)
        raw[0x80:0x82] = b"O1"
        struct.pack_into("<f", raw, 0x80 + 0x18, 0.5)
        struct.pack_into("<hhhh", raw, 0xC0, 10, 20, 30, 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rest.eeg"
            path.write_bytes(raw)
            recording = load_recording(path)
        self.assertEqual(recording.source_format, "wineeg")
        self.assertEqual(recording.ch_names, ["O1"])
        self.assertEqual(recording.sfreq, 256.0)
        self.assertEqual(recording.data.shape, (1, 4))
        self.assertTrue(np.allclose(recording.data[0], [5.0, 10.0, 15.0, 20.0]))
        self.assertEqual(recording.marks, [])

    def test_wineeg_marks_split_eyes_closed_and_open(self):
        raw = bytearray(0x900)
        struct.pack_into("<H", raw, 0x00, 0x0F0E)
        struct.pack_into("<H", raw, 0x02, 1)
        struct.pack_into("<H", raw, 0x04, 256)
        struct.pack_into("<H", raw, 0x06, 2)
        struct.pack_into("<H", raw, 0x0A, 2)
        struct.pack_into("<I", raw, 0x10, 1024)
        struct.pack_into("<I", raw, 0x18, 0x80)
        struct.pack_into("<I", raw, 0x1C, 0xC0)
        struct.pack_into("<I", raw, 0x20, 0x8C0)
        raw[0x80:0x82] = b"O1"
        struct.pack_into("<f", raw, 0x80 + 0x18, 1.0)
        _pack_mark(raw, 0x8C0, 0, "Eyes closed")
        _pack_mark(raw, 0x8E0, 2000, "Eyes open")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "both.eeg"
            path.write_bytes(raw)
            recording = load_recording(path)
        self.assertEqual(len(recording.marks), 2)
        self.assertIn("Eyes closed", recording.notes[-1])
        sides = split_marked_conditions(recording)
        self.assertEqual(set(sides), {"ec", "eo"})
        self.assertAlmostEqual(sides["ec"].duration_s, 2.0, delta=0.02)
        self.assertAlmostEqual(sides["eo"].duration_s, 2.0, delta=0.02)
        self.assertIn("Eyes closed mark", sides["ec"].notes[-1])

    def test_wineeg_clock_mark_does_not_cut_the_file(self):
        raw = bytearray(0x8E0)
        struct.pack_into("<H", raw, 0x00, 0x0F0E)
        struct.pack_into("<H", raw, 0x02, 1)
        struct.pack_into("<H", raw, 0x04, 256)
        struct.pack_into("<H", raw, 0x06, 1)
        struct.pack_into("<H", raw, 0x0A, 2)
        struct.pack_into("<I", raw, 0x10, 1024)
        struct.pack_into("<I", raw, 0x18, 0x80)
        struct.pack_into("<I", raw, 0x1C, 0xC0)
        struct.pack_into("<I", raw, 0x20, 0x8C0)
        raw[0x80:0x82] = b"O1"
        struct.pack_into("<f", raw, 0x80 + 0x18, 1.0)
        _pack_mark(raw, 0x8C0, 0, " 10:36:38")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clock.eeg"
            path.write_bytes(raw)
            recording = load_recording(path)
        self.assertEqual(split_marked_conditions(recording), {})
        self.assertAlmostEqual(recording.duration_s, 4.0, delta=0.02)
        self.assertIn("10:36:38", " ".join(recording.notes))

    def test_csv_and_json_roundtrip(self):
        times, data, names = _synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "recording.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("time_s," + ",".join(names) + "\n")
                for index, moment in enumerate(times):
                    values = ",".join(f"{sample:.6f}" for sample in data[:, index])
                    handle.write(f"{moment:.6f},{values}\n")
            self.assertFalse(needs_sfreq(csv_path))
            recording = load_recording(csv_path)
            self.assertEqual(recording.ch_names, ["Fp1", "F3", "Cz"])
            self.assertAlmostEqual(recording.sfreq, 256.0, delta=0.5)
            self.assertAlmostEqual(float(np.max(np.abs(recording.data[0]))), 20.0, delta=0.1)

            bare = root / "notime.csv"
            bare.write_text("Fp1\n1\n2\n", encoding="utf-8")
            self.assertTrue(needs_sfreq(bare))
            with self.assertRaises(ValueError):
                load_recording(bare)

            embedded = root / "embedded.json"
            embedded.write_text(
                '{"sfreq": 256, "unit": "V", "channel_names": ["Cz"], "data": [[1e-5, -1e-5]]}',
                encoding="utf-8",
            )
            volts = load_recording(embedded)
            self.assertEqual(volts.ch_names, ["Cz"])
            self.assertAlmostEqual(float(volts.data[0, 0]), 10.0, places=4)


def _recording():
    cached = getattr(_recording, "value", None)
    if cached is not None:
        return cached
    times, data, names = _synthetic()
    root = tempfile.TemporaryDirectory()
    csv_path = Path(root.name) / "fit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("time_s," + ",".join(names) + "\n")
        for index, moment in enumerate(times):
            values = ",".join(f"{sample:.6f}" for sample in data[:, index])
            handle.write(f"{moment:.6f},{values}\n")
    _recording.root = root
    _recording.value = load_recording(csv_path)
    return _recording.value


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = _recording()

    @classmethod
    def tearDownClass(cls):
        root = getattr(_recording, "root", None)
        if root is not None:
            root.cleanup()
            _recording.root = None
            _recording.value = None

    def test_specparam_finds_alpha(self):
        fit = fit_spectra(self.recording)
        by_name = {item.name: item for item in fit.channels}
        self.assertFalse(by_name["F3"].usable)
        peak = by_name["Fp1"].dominant_peak
        self.assertIsNotNone(peak)
        self.assertGreater(peak[0], 8.0)
        self.assertLess(peak[0], 12.0)
        self.assertIsNotNone(by_name["Fp1"].exponent)
        self.assertGreater(by_name["Fp1"].r_squared or 0, 0.5)
        self.assertEqual(fit.aperiodic_mode, "fixed")
        self.assertAlmostEqual(fit.t_start, 0.0)
        self.assertAlmostEqual(fit.t_end, self.recording.duration_s, delta=0.05)

    def test_fit_window_and_range(self):
        settings = FitSettings(t_start=1.0, t_length=4.0, fmin=2.0, fmax=30.0, segment_s=2.0)
        fit = fit_spectra(self.recording, settings)
        self.assertAlmostEqual(fit.fmin, 2.0)
        self.assertGreater(fit.fmax, 20.0)
        self.assertLessEqual(fit.fmax, 30.0)
        self.assertAlmostEqual(fit.t_start, 1.0, delta=0.02)
        self.assertAlmostEqual(fit.t_end, 5.0, delta=0.05)
        self.assertAlmostEqual(fit.segment_s, 2.0)

    def test_alpha_band_can_move(self):
        item = ChannelFit(name="O1", index=0, usable=True, peaks=[(6.2, 1.0, 1.0), (10.1, 0.4, 1.0)])
        self.assertAlmostEqual(alpha_peak(item)[0], 10.1)
        self.assertAlmostEqual(alpha_peak(item, (4.0, 8.0))[0], 6.2)
        self.assertEqual(band_label(8.0, 12.0), "Alpha")
        self.assertEqual(band_label(4.0, 8.0), "Theta")
        self.assertEqual(band_label(7.0, 13.0), "7–13 Hz")
        self.assertEqual(band_phrase(13.0, 30.0), "Beta 13–30 Hz")
        self.assertEqual(band_phrase(7.0, 13.0), "7–13 Hz")
        exported = comparison_rows([ChannelComparison(name="O1", alpha_reactivity=1.0)], (4.0, 8.0))
        self.assertEqual(exported[0]["band"], "Theta")
        self.assertEqual(exported[0]["band_low_hz"], 4.0)
        self.assertEqual(exported[0]["band_high_hz"], 8.0)
        self.assertEqual(exported[0]["alpha_reactivity_ec_minus_eo"], 1.0)

    def test_bycycle_alpha_period(self):
        cycles = fit_cycles(self.recording, 0, (8.0, 12.0))
        self.assertGreater(cycles.n_cycles, 20)
        self.assertIsNotNone(cycles.period_ms)
        self.assertAlmostEqual(cycles.period_ms, 100.0, delta=15.0)
        self.assertIsNotNone(cycles.rise_decay)
        self.assertIsNotNone(cycles.volt_amp)
        self.assertGreater(cycles.volt_amp or 0, 1.0)
        if cycles.n_bursts == 0 and cycles.n_cycles:
            self.assertTrue(cycles.burst_note.startswith("No burst"))
        else:
            self.assertEqual(cycles.burst_note, "")
        self.assertEqual(cycle_summary_row(cycles, "ec")["burst_note"], cycles.burst_note)
        self.assertEqual(len(cycles.cutoffs), 4)
        self.assertEqual(len(cycles.episodes), cycles.n_burst_episodes)
        self.assertEqual(sum(episode.n_cycles for episode in cycles.episodes), cycles.n_burst_cycles)
        summary = cycle_summary_row(cycles, "ec")
        self.assertIn("monotonicity_median", summary)
        self.assertEqual(summary["monotonicity_n_passed"], next(
            cutoff.n_passed for cutoff in cycles.cutoffs if cutoff.key == "monotonicity"
        ))
        self.assertEqual(len(burst_episode_rows(cycles, "ec")), cycles.n_burst_episodes)

    def test_burst_note_names_the_cutoff(self):
        thresholds = FitSettings().cycle_thresholds()
        quiet = _frame({
            "amp_fraction": [1, 1, 1, 1, 1],
            "amp_consistency": [1, 1, 1, 1, 1],
            "period_consistency": [1, 1, 1, 1, 1],
            "monotonicity": [0.9, 0.2, 0.2, 0.2, 0.9],
        })
        self.assertIn("monotonicity above 0.8", burst_note(quiet, thresholds))
        exact = _frame({
            "amp_fraction": [1, 1, 1, 1, 1],
            "amp_consistency": [1, 1, 1, 1, 1],
            "period_consistency": [1, 1, 1, 1, 1],
            "monotonicity": [0.8, 0.8, 0.8, 0.8, 0.8],
        })
        self.assertIn("monotonicity above 0.8", burst_note(exact, thresholds))
        short = _frame({
            "amp_fraction": [0, 1, 0, 1, 0],
            "amp_consistency": [1, 1, 1, 1, 1],
            "period_consistency": [1, 1, 1, 1, 1],
            "monotonicity": [1, 1, 1, 1, 1],
        })
        note = burst_note(short, thresholds)
        self.assertIn("longest run is 1", note)
        self.assertIn("needs 3", note)

    def test_burst_cutoffs_count_strict_passes(self):
        thresholds = FitSettings().cycle_thresholds()
        frame = _frame({
            "amp_fraction": [1, 1, 1, 1, 1],
            "amp_consistency": [1, 1, 1, 1, 1],
            "period_consistency": [1, 1, 1, 1, 1],
            "monotonicity": [0.9, 0.8, 0.95, 0.2, 0.9],
        })
        rows = {row.key: row for row in burst_cutoffs(frame, thresholds)}
        self.assertEqual(rows["monotonicity"].n_passed, 3)
        self.assertEqual(rows["monotonicity"].n_cycles, 5)
        self.assertEqual(rows["amp_fraction"].n_passed, 5)
        self.assertAlmostEqual(rows["monotonicity"].median, 0.9)

    def test_burst_episodes_stay_separate(self):
        frame = _frame({
            "is_burst": [False, True, True, True, False, True, True, False],
            "sample_last_trough": [0, 10, 20, 30, 40, 50, 60, 70],
            "sample_next_trough": [10, 20, 30, 40, 50, 60, 70, 80],
            "period": [10, 10, 10, 10, 10, 12, 12, 10],
            "volt_amp": [1, 2, 4, 6, 1, 3, 5, 1],
        })
        episodes = burst_episodes(frame, 100.0, t_offset=5.0)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].n_cycles, 3)
        self.assertAlmostEqual(episodes[0].start_s, 5.1)
        self.assertAlmostEqual(episodes[0].end_s, 5.4)
        self.assertAlmostEqual(episodes[0].volt_amp, 4.0)
        self.assertEqual(episodes[1].n_cycles, 2)
        self.assertAlmostEqual(episodes[1].period_ms, 120.0)
        self.assertEqual(burst_episodes(frame, 0.0), [])


class QualityTests(unittest.TestCase):
    def test_voltage_scale_is_named_and_not_rescaled(self):
        self.assertIsNone(voltage_scale_note(np.full((1, 40), 25.0)))
        self.assertIn("microvolts", voltage_scale_note(np.full((1, 40), 1e-5)) or "")
        self.assertIn("above", voltage_scale_note(np.full((1, 40), 50000.0)) or "")
        recording = finalize(
            np.vstack([np.full(64, 1e-5), np.full(64, 2.0), np.full(64, 4.0)]),
            ["O1", "A1", "A2"], 256.0, "scale.edf", "edf",
        )
        self.assertEqual(recording.ch_names, ["O1"])
        self.assertIn("A1", recording.reference_channels)
        self.assertIn("A2", recording.reference_channels)
        self.assertTrue(any("linked-ear" in note for note in recording.notes))
        self.assertTrue(any("microvolts" in note for note in recording.notes))
        self.assertAlmostEqual(float(recording.data[0, 0]), 1e-5)

    def test_large_samples_stay_out_of_the_fit(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        signal = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        signal[(times >= 3.0) & (times < 4.0)] = 5000.0
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="loud.csv", source_format="csv",
        )
        prepared = prepare_recording(recording, FitSettings())
        self.assertTrue(np.isnan(prepared.data[0, int(3 * sfreq):int(4 * sfreq)]).all())
        self.assertTrue(np.isfinite(prepared.data[0, : int(2 * sfreq)]).all())
        self.assertTrue(any("500" in note for note in prepared.notes))
        fit = fit_spectra(recording)
        peak = fit.channels[0].dominant_peak
        self.assertIsNotNone(peak)
        self.assertGreater(peak[0], 8.0)
        self.assertLess(peak[0], 12.0)
        self.assertIn("500", fit.notes)
        loud = np.full_like(signal, 5000.0)
        both = Recording(
            data=np.vstack([signal, loud]), ch_names=["O1", "Fp1"], sfreq=sfreq,
            source_path="loud.csv", source_format="csv",
        )
        both_fit = fit_spectra(both)
        reasons = {item.name: item.reason for item in both_fit.channels}
        self.assertEqual(reasons["Fp1"], "Amplitude above the limit.")
        self.assertTrue(both_fit.channels[0].name == "O1" and both_fit.channels[0].usable)

    def test_a_spike_does_not_enter_the_average(self):
        values = np.full(16, 10.0)
        spike = values.copy()
        spike[4] = 5000.0
        recording = Recording(
            data=np.vstack([spike, values]), ch_names=["Fp1", "O1"], sfreq=256.0,
            source_path="spike.csv", source_format="csv",
        )
        prepared = prepare_recording(recording, FitSettings(reference="average"))
        self.assertTrue(np.isnan(prepared.data[0, 4]))
        self.assertAlmostEqual(float(prepared.data[1, 4]), 10.0)
        self.assertAlmostEqual(float(prepared.data[0, 0]), 0.0)
        self.assertAlmostEqual(float(prepared.data[1, 0]), 0.0)

    def test_a_short_window_is_not_called_an_epoch(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        signal = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="short.csv", source_format="csv",
        )
        fit = fit_spectra(recording, FitSettings(epoch_s=30.0))
        self.assertIsNone(fit.epoch_s)
        self.assertEqual(fit.epoch_count, 0)
        self.assertIn("shorter than one", fit.notes)

    def test_reference_choices(self):
        data = np.vstack([np.full(8, 10.0), np.zeros(8)])
        recording = Recording(
            data=data, ch_names=["Fp1", "O1"], sfreq=256.0,
            source_path="ref.csv", source_format="csv",
            reference_channels={"A1": np.full(8, 10.0), "A2": np.full(8, 20.0)},
        )
        averaged = apply_reference(recording, "average")
        self.assertAlmostEqual(float(averaged.data[0, 0]), 5.0)
        self.assertAlmostEqual(float(averaged.data[1, 0]), -5.0)
        linked = apply_reference(recording, "linked")
        self.assertAlmostEqual(float(linked.data[0, 0]), -5.0)
        stored = apply_reference(recording, "stored")
        self.assertIs(stored, recording)
        missing = apply_reference(
            Recording(data=data, ch_names=["Fp1", "O1"], sfreq=256.0, source_path="x", source_format="csv"),
            "linked",
        )
        self.assertIn("not in the file", missing.notes[-1])
        self.assertAlmostEqual(float(missing.data[0, 0]), 10.0)

    def test_nearest_peak_stays_out_of_the_band_number(self):
        outside = ChannelFit(name="C3", index=0, usable=True, peaks=[(12.4, 0.2, 1.0), (6.5, 0.4, 1.0)])
        found = nearest_outside_peak(outside, (8.0, 12.0))
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found[0], 12.4)
        self.assertAlmostEqual(found[1], 0.4)
        inside = ChannelFit(name="O1", index=1, usable=True, peaks=[(10.0, 1.0, 1.0)])
        self.assertIsNone(nearest_outside_peak(inside, (8.0, 12.0)))
        self.assertTrue(peak_at_width_floor(0.25, 0.5))
        self.assertFalse(peak_at_width_floor(1.0, 0.5))
        rows = compare_spectra(
            _one_fit(outside),
            _one_fit(inside),
        )
        by_name = {row.name: row for row in rows}
        self.assertIsNone(by_name["C3"].alpha_reactivity)
        self.assertAlmostEqual(by_name["C3"].nearest_cf_ec, 12.4)
        self.assertIsNone(by_name["O1"].nearest_cf_ec)

    def test_whole_file_comparison_uses_the_shorter_length(self):
        long = Recording(
            data=np.zeros((1, 2560)), ch_names=["O1"], sfreq=256.0,
            source_path="long.csv", source_format="csv",
        )
        short = Recording(
            data=np.zeros((1, 1024)), ch_names=["O1"], sfreq=256.0,
            source_path="short.csv", source_format="csv",
        )
        whole = FitSettings()
        self.assertAlmostEqual(matched_settings(whole, long, [short]).t_length, short.duration_s)
        self.assertIsNone(matched_settings(whole, short, [long]).t_length)
        explicit = FitSettings(t_length=2.0)
        self.assertAlmostEqual(matched_settings(explicit, long, [short]).t_length, 2.0)

    def test_epochs_count_a_peak_only_where_it_is_present(self):
        sfreq = 256.0
        times = np.arange(int(16.0 * sfreq)) / sfreq
        signal = np.where(
            times < 8.0,
            20.0 * np.sin(2 * np.pi * 6.0 * times),
            20.0 * np.sin(2 * np.pi * 10.0 * times),
        )
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="epochs.csv", source_format="csv",
        )
        fit = fit_spectra(recording, FitSettings(epoch_s=8.0))
        item = fit.channels[0]
        self.assertEqual(fit.epoch_count, 2)
        self.assertEqual(item.n_epochs, 2)
        self.assertEqual(item.n_peak_epochs, 1)
        self.assertTrue(any(8.0 <= peak[0] <= 12.0 for peak in item.peaks))

    def test_a_short_tail_is_left_out_of_the_epochs(self):
        sfreq = 256.0
        times = np.arange(int(20.0 * sfreq)) / sfreq
        signal = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="tail.csv", source_format="csv",
        )
        fit = fit_spectra(recording, FitSettings(epoch_s=8.0))
        self.assertEqual(fit.epoch_count, 2)
        self.assertIn("last 4", fit.notes)

    def test_one_epoch_does_not_merge_its_own_nearby_peaks(self):
        from bifooof.analysis.specparam_fit import _median_peaks

        grouped = _median_peaks([
            [(10.0, 1.0, 1.0), (10.4, 0.2, 1.0)],
            [(10.2, 0.8, 1.0)],
        ])
        self.assertEqual(len(grouped), 2)
        heights = sorted(peak[1] for peak in grouped)
        self.assertAlmostEqual(heights[0], 0.2)
        self.assertAlmostEqual(heights[1], 0.9)

    def test_band_change_keeps_each_peaks_epoch_count(self):
        from bifooof.analysis.specparam_fit import SpectrumFit, epochs_for_peak

        alpha = (10.0, 0.8, 0.25)
        theta = (6.0, 0.4, 2.0)
        closed = ChannelFit(
            name="O1", index=0, usable=True, peaks=[alpha, theta],
            peak_counts=[1, 2], n_epochs=2, n_peak_epochs=1, peak_at_width_floor=True,
            exponent=1.0, r_squared=0.9,
        )
        opened = ChannelFit(
            name="O1", index=0, usable=True, peaks=[theta], exponent=1.1, r_squared=0.9,
        )
        self.assertEqual(epochs_for_peak(closed, theta), 2)
        self.assertEqual(epochs_for_peak(closed, alpha), 1)
        rows = compare_spectra(
            SpectrumFit(channels=[closed], fmin=1.0, fmax=40.0),
            SpectrumFit(channels=[opened], fmin=1.0, fmax=40.0),
            (4.0, 8.0),
            0.5,
        )
        self.assertFalse(rows[0].peak_at_floor_ec)
        self.assertAlmostEqual(rows[0].alpha_cf_ec, 6.0)

    def test_findings_keep_a_peak_separate_from_the_background(self):
        from bifooof.analysis.cycles import CycleFit
        from bifooof.analysis.findings import compile_report, report_text
        from bifooof.analysis.specparam_fit import SpectrumFit

        def channel(name, index, peaks, exponent, r_squared, level):
            item = ChannelFit(
                name=name, index=index, usable=True, exponent=exponent,
                r_squared=r_squared, peaks=list(peaks),
            )
            item.freqs = np.linspace(1.0, 40.0, 40)
            item.spectrum = np.full(40, level)
            return item

        closed = SpectrumFit(channels=[
            channel("O1", 0, [(10.0, 0.80, 1.5)], 1.2, 0.92, 1.0),
            channel("C3", 1, [(12.4, 0.20, 1.0), (6.0, 0.30, 1.0)], 0.9, 0.40, 1.0),
        ], fmin=1.0, fmax=40.0, t_start=0.0, t_end=60.0)
        opened = SpectrumFit(channels=[
            channel("O1", 0, [(9.0, 0.20, 1.2)], 1.4, 0.88, 0.4),
            channel("C3", 1, [], 1.0, 0.80, 0.4),
        ], fmin=1.0, fmax=40.0, t_start=0.0, t_end=60.0)
        closed.epoch_s = 30.0
        closed.epoch_count = 2
        closed.channels[1].n_epochs = 2
        closed.channels[1].n_peak_epochs = 1
        cycles = {"ec": CycleFit(
            channel="C3", channel_index=1, band=(8.0, 12.0), sfreq=256.0,
            filtered=np.zeros(8), n_cycles=10, n_burst_cycles=6, n_burst_episodes=2,
            burst_duration_s=1.2, burst_fraction=0.3, period_ms=100.0, rise_decay=0.48,
            peak_trough=0.51, volt_amp=18.0, burst_period_ms=98.0, burst_rise_decay=0.47,
            burst_peak_trough=0.52, burst_volt_amp=22.0,
        )}
        recording = Recording(
            data=np.zeros((2, 8)), ch_names=["O1", "C3"], sfreq=256.0,
            source_path="eyes.eeg", source_format="eeg",
            notes=["Voltage looks like volts rather than microvolts. Peak height and burst amplitude assume microvolts."],
        )
        text = report_text(compile_report(
            {"ec": recording, "eo": recording},
            {"ec": closed, "eo": opened},
            cycles, (8.0, 12.0), "C3", FitSettings(),
        ))
        self.assertIn("10.00 Hz", text)
        self.assertIn("aperiodic background", text)
        self.assertIn("Exponent 0.90", text)
        self.assertIn("12.40 Hz", text)
        self.assertIn("Theta", text)
        self.assertIn("weak", text.lower())
        self.assertIn("volts rather than microvolts", text)
        self.assertIn("higher with eyes closed", text)
        self.assertIn("no peak difference", text)
        self.assertIn("separate runs", text)
        self.assertIn("raw EEG", text)
        self.assertIn("share of total spectral power", text)
        self.assertIn("not entered as a height of zero", text)
        self.assertIn("does not scale the height", text)
        self.assertIn("Reference: as stored", text)
        for label in ("ADHD", "hyperarousal", "Alzheimer"):
            self.assertNotIn(label, text)
        renamed = report_text(compile_report(
            {"ec": recording, "eo": recording},
            {"ec": closed, "eo": opened},
            cycles, (8.0, 12.0), "C3", FitSettings(),
            pair=("Pre", "Post"),
        ))
        self.assertIn("Pre and Post", renamed)
        self.assertIn("higher with Pre", renamed)
        self.assertIn("Post minus Pre", renamed)
        self.assertIn("(Pre − Post) / Post", renamed)
        self.assertNotIn("eyes closed", renamed.lower())
        self.assertNotIn("eyes open", renamed.lower())
        custom = report_text(compile_report(
            {"ec": recording, "eo": recording},
            {"ec": closed, "eo": opened},
            cycles, (8.0, 12.0), "C3", FitSettings(),
            pair=("Baseline", "Follow-up"),
        ))
        self.assertIn("Baseline and Follow-up", custom)
        self.assertIn("higher with Baseline", custom)
        self.assertNotIn("eyes closed", custom.lower())


def _one_fit(channel: ChannelFit):
    from bifooof.analysis.specparam_fit import SpectrumFit

    channel.freqs = np.linspace(1.0, 40.0, 16)
    channel.spectrum = np.zeros(16)
    return SpectrumFit(channels=[channel], fmin=1.0, fmax=40.0, t_start=0.0, t_end=8.0)


class ExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = _recording()

    @classmethod
    def tearDownClass(cls):
        root = getattr(_recording, "root", None)
        if root is not None:
            root.cleanup()
            _recording.root = None
            _recording.value = None

    def test_welch_removes_a_straight_drift(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        tone = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        drifted = tone + np.linspace(0.0, 20000.0, tone.size)
        freqs, clean, _nperseg = welch_spectra(tone[None, :], sfreq)
        _, dirty, _ignored = welch_spectra(drifted[None, :], sfreq)
        peak = int(np.argmin(np.abs(freqs - 10.0)))
        self.assertLess(abs(clean[0, peak] - dirty[0, peak]) / clean[0, peak], 0.05)
        self.assertLess(dirty[0, 0], clean[0, peak])

    def test_nonfinite_samples_stay_excluded(self):
        recording = finalize(np.array([[1.0, np.nan, 3.0, 4.0]]), ["Cz"], 256.0, "gap.csv", "csv")
        self.assertTrue(np.isnan(recording.data[0, 1]))
        self.assertIsNotNone(recording.valid)
        self.assertFalse(bool(recording.valid[0, 1]))
        self.assertTrue(bool(recording.valid[0, 0]))
        self.assertNotIn("0.0", " ".join(f"{value:.1f}" for value in recording.data[0] if np.isfinite(value)))

    def test_bad_annotation_is_left_out_of_the_fit(self):
        import mne

        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        signal = 20e-6 * np.sin(2 * np.pi * 10.0 * times)
        signal = signal.copy()
        gap = (times >= 3.0) & (times < 4.0)
        signal[gap] = 500e-6 * np.sin(2 * np.pi * 25.0 * times[gap])
        raw = mne.io.RawArray(
            signal[None, :], mne.create_info(["Cz"], sfreq, "eeg"), verbose="ERROR",
        )
        raw.set_annotations(mne.Annotations(onset=[3.0], duration=[1.0], description=["BAD_artifact"]))
        recording = _recording_from_raw(raw, Path("marked.fif"), "fif")
        i0, i1 = int(3.0 * sfreq), int(4.0 * sfreq)
        self.assertTrue(np.isnan(recording.data[0, i0:i1]).all())
        self.assertFalse(bool(recording.valid[0, i0:i1].any()))
        self.assertTrue(any("BAD" in note for note in recording.notes))
        fit = fit_spectra(recording)
        peak = fit.channels[0].dominant_peak
        self.assertIsNotNone(peak)
        self.assertGreater(peak[0], 8.0)
        self.assertLess(peak[0], 12.0)
        self.assertTrue(all(not (24.0 <= center <= 26.0) for center, _height, _width in fit.channels[0].peaks))
        cycles = fit_cycles(recording, 0, (8.0, 12.0))
        for start, stop, _is_burst in cycles.spans:
            self.assertTrue(stop <= i0 or start >= i1)

    def test_bad_channel_is_not_fit(self):
        import mne

        sfreq = 256.0
        times = np.arange(int(4.0 * sfreq)) / sfreq
        data = np.vstack([
            20e-6 * np.sin(2 * np.pi * 10.0 * times),
            20e-6 * np.sin(2 * np.pi * 6.0 * times),
        ])
        raw = mne.io.RawArray(data, mne.create_info(["Cz", "O1"], sfreq, "eeg"), verbose="ERROR")
        raw.info["bads"] = ["Cz"]
        recording = _recording_from_raw(raw, Path("bads.fif"), "fif")
        self.assertEqual(recording.bad_channels, ["Cz"])
        self.assertFalse(bool(recording.valid[recording.ch_names.index("Cz")].any()))
        fit = fit_spectra(recording)
        by_name = {item.name: item for item in fit.channels}
        self.assertFalse(by_name["Cz"].usable)
        self.assertEqual(by_name["Cz"].reason, "Bad channel.")
        self.assertTrue(by_name["O1"].usable)

    def test_single_channel_fallback_does_not_pass_group_arguments(self):
        import specparam

        group_fit = specparam.SpectralGroupModel.fit
        model_fit = specparam.SpectralModel.fit
        calls: list[dict] = []

        def fail(self, *args, **kwargs):
            raise RuntimeError("group fit forced to fail")

        def spy(self, *args, **kwargs):
            calls.append(dict(kwargs))
            return model_fit(self, *args, **kwargs)

        specparam.SpectralGroupModel.fit = fail
        specparam.SpectralModel.fit = spy
        try:
            fit = fit_spectra(self.recording)
        finally:
            specparam.SpectralGroupModel.fit = group_fit
            specparam.SpectralModel.fit = model_fit
        self.assertTrue(calls)
        self.assertNotIn("n_jobs", calls[0])
        self.assertNotIn("progress", calls[0])
        self.assertIsNotNone(next(item.exponent for item in fit.channels if item.name == "Fp1"))

    def test_burst_episodes_are_separate_from_burst_cycles(self):
        self.assertEqual(episode_count(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)), 2)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_shows_fit(self):
        window = MainWindow()
        window.show_recording(_recording())
        deadline = time.time() + 60
        while time.time() < deadline:
            self.app.processEvents()
            spectrum_done = window.spectrum.fit is not None or "fail" in window.statusBar().currentMessage().lower()
            cycle_done = window.cycles.fit is not None or "above the Nyquist" in window.cycles.caption.text()
            if window.spectrum._worker is not None and not window.spectrum._worker.isRunning():
                spectrum_done = True
            if window.cycles._worker is not None and not window.cycles._worker.isRunning():
                cycle_done = cycle_done or window.cycles.fit is not None or window.cycles.caption.text() != "Finding cycles on Fp1…"
            if spectrum_done and cycle_done and window.spectrum.fit is not None and window.cycles.fit is not None:
                break
            time.sleep(0.05)
        self.app.processEvents()
        self.assertIsNotNone(window.spectrum.fit, window.statusBar().currentMessage())
        self.assertIsNotNone(window.cycles.fit, window.cycles.caption.text())
        self.assertFalse(window.windowIcon().isNull())
        self.assertIn("Eyes closed", window.spectrum.caption.text())
        self.assertEqual(window.trace.channels.count(), 3)
        window.trace.window.setCurrentIndex(0)
        window.trace.gain.setCurrentIndex(1)
        from PySide6.QtCore import Qt
        window.trace.channels.item(1).setCheckState(Qt.Unchecked)
        self.app.processEvents()
        self.assertEqual(window.trace._win_s, 5.0)
        self.assertNotIn(1, window.trace._visible)
        window.trace.channel_selected.emit(2)
        self.app.processEvents()
        self.assertEqual(window.spectrum._selected, 2)
        window.cycles.band.setCurrentIndex(0)
        if window.cycles._worker is not None:
            window.cycles._worker.wait(30000)
        self.app.processEvents()
        self.assertIsNotNone(window.cycles.fit)
        self.assertEqual(window.cycles.fit.band, (1.0, 4.0))
        self.assertIn("accepted time", window.cycles.caption.text())
        self.assertEqual(window.cycles.cutoffs.item(0, 0).text(), "Amplitude fraction")
        self.assertIn("/", window.cycles.cutoffs.item(0, 3).text())
        low, high = window.fit_bar.current_band()
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 4.0)
        self.assertTrue(window.fit_bar.band.currentText().startswith("Delta"))
        self.assertIn("Findings", window.tabs.tabText(window.tabs.indexOf(window.findings)))
        self.assertIn("Eyes closed", window.findings.plain)
        self.assertIn("raw EEG", window.findings.plain)
        self.assertNotIn("ADHD", window.findings.plain)
        self.assertNotIn("hyperarousal", window.findings.plain)
        for fig in (window.trace.canvas.fig, window.spectrum.canvas.fig, window.cycles.canvas.fig):
            self.assertTrue(any(text.get_text() == "EEG Paradox Solutions" for text in fig.texts))
        window.close()

    def test_marks_open_both_sides(self):
        recording = _tone(20.0, "rest.eeg")
        recording.marks = [Mark(0.0, "Eyes closed"), Mark(4.0, "Eyes open")]
        window = MainWindow()
        window._present(recording, "eo")
        self.app.processEvents()
        self.assertEqual(window.active, "eo")
        self.assertAlmostEqual(window.recordings["ec"].duration_s, 4.0, delta=0.05)
        self.assertAlmostEqual(window.recordings["eo"].duration_s, 4.0, delta=0.05)
        self.assertIn("0.0–4.0 s", window.file_label.text())
        self.assertIn("4.0–8.0 s", window.file_label.text())
        window.close()


def _tone(scale: float, name: str) -> Recording:
    sfreq = 256.0
    times = np.arange(int(8.0 * sfreq)) / sfreq
    fp1 = scale * np.sin(2 * np.pi * 10.0 * times)
    flat = np.zeros_like(times)
    cz = 8.0 * np.sin(2 * np.pi * 6.0 * times)
    return Recording(
        data=np.vstack([fp1, flat, cz]),
        ch_names=["Fp1", "F3", "Cz"],
        sfreq=sfreq,
        source_path=name,
        source_format="edf",
    )


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_condition_names(self):
        self.assertEqual(guess_condition(r"C:\data\LeMac-EC-0622.edf"), "ec")
        self.assertEqual(guess_condition("eyes closed.edf"), "ec")
        self.assertEqual(guess_condition("subject_EO.edf"), "eo")
        self.assertEqual(guess_condition("eyes-open.edf"), "eo")
        self.assertEqual(guess_condition("EC1.edf"), "ec")
        self.assertEqual(guess_condition("EO2.edf"), "eo")
        self.assertEqual(guess_condition("eyesclosed.edf"), "ec")
        self.assertIsNone(guess_condition("recording.edf"))
        self.assertIsNone(guess_condition("EOG.edf"))
        self.assertIsNone(guess_condition("echo.edf"))

    def test_pair_comparison_and_session_save(self):
        window = MainWindow()
        window.show_recording(_tone(20.0, "EC.edf"), "ec")
        window.show_recording(_tone(5.0, "EO.edf"), "eo")
        deadline = time.time() + 40
        while time.time() < deadline:
            self.app.processEvents()
            if "ec" in window.spectrum.fits and "eo" in window.spectrum.fits and window.compare.ready:
                break
            time.sleep(0.05)
        self.app.processEvents()
        self.assertTrue(window.compare.ready, window.statusBar().currentMessage())
        self.assertTrue(any(
            text.get_text() == "EEG Paradox Solutions" for text in window.compare.canvas.fig.texts
        ))
        fp1 = next(row for row in window.compare.rows if row.name == "Fp1")
        self.assertIsNotNone(fp1.alpha_cf_ec)
        self.assertIsNotNone(fp1.alpha_cf_eo)
        self.assertGreater(fp1.offset_ec - fp1.offset_eo, 0.2)
        self.assertEqual(window.active, "eo")
        window.view.setCurrentIndex(window.view.findData("ec"))
        self.app.processEvents()
        self.assertEqual(window.active, "ec")
        self.assertIn("Eyes closed", window.file_label.text())
        self.assertIn("Eyes open", window.file_label.text())
        from PySide6.QtCore import Qt
        window.trace.channels.item(1).setCheckState(Qt.Unchecked)
        window.view.setCurrentIndex(window.view.findData("eo"))
        self.app.processEvents()
        window.view.setCurrentIndex(window.view.findData("ec"))
        self.app.processEvents()
        self.assertEqual(window.active, "ec")
        self.assertEqual(window.trace.channels.item(1).checkState(), Qt.Unchecked)
        self.assertIn("Eyes closed", window.trace.canvas.fig.axes[0].get_title(loc="left"))
        for fig in (window.trace.canvas.fig, window.spectrum.canvas.fig):
            self.assertTrue(any(text.get_text() == "EEG Paradox Solutions" for text in fig.texts))
        with tempfile.TemporaryDirectory() as tmp:
            written = window.save_session(tmp)
            names = {path.name for path in written}
            self.assertIn("ec_spectrum.csv", names)
            self.assertIn("eo_spectrum.csv", names)
            self.assertIn("comparison.csv", names)
            self.assertIn("comparison.png", names)
            self.assertIn("ec_spectrum.png", names)
            self.assertIn("eo_spectrum.png", names)
            self.assertIn("traces.png", names)
            text = Path(tmp, "comparison.csv").read_text(encoding="utf-8")
            self.assertIn("Fp1", text)
            self.assertIn("alpha_reactivity_ec_minus_eo", text)
            self.assertIn("peak_height_definition", text)
            self.assertIn("band_power_reactivity_percent", text)
            self.assertIn("band_low_hz", text)
            self.assertIn("band_high_hz", text)
            metadata = Path(tmp, "ec_metadata.json").read_text(encoding="utf-8")
            self.assertIn("microvolts", metadata)
            self.assertIn("linear", metadata)
            self.assertIn("n_burst_cycles", metadata)
            self.assertTrue(Path(tmp, "ec_peaks.csv").is_file())
        closed, opened = window.spectrum.fits["ec"], window.spectrum.fits["eo"]
        again = compare_spectra(closed, opened)
        self.assertEqual(again[0].name, "Fp1")
        window.close()

    def test_app_icon(self):
        from bifooof.ui.icons import app_icon, pixmap, save_app_icon

        self.assertFalse(pixmap("logo", "#1c4e78", 32).isNull())
        self.assertFalse(app_icon().isNull())
        with tempfile.TemporaryDirectory() as tmp:
            ico = save_app_icon(tmp)
            self.assertEqual(ico.read_bytes()[:4], b"\x00\x00\x01\x00")
            self.assertGreater((Path(tmp) / "bifooof-256.png").stat().st_size, 100)

    def test_cli_info_and_run(self):
        import subprocess
        import sys

        recording = _recording()
        source = recording.source_path
        with tempfile.TemporaryDirectory() as tmp:
            info = subprocess.run(
                [sys.executable, "-m", "bifooof", "info", source],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(info.returncode, 0, info.stderr)
            self.assertIn("Fp1", info.stdout)
            self.assertIn("256", info.stdout)
            out = Path(tmp)
            run = subprocess.run(
                [sys.executable, "-m", "bifooof", "run", source, "--condition", "ec",
                 "--channel", "Fp1", "--out", str(out), "--no-cycles"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            folder = out / "fit"
            self.assertTrue((folder / "spectrum.csv").is_file(), run.stdout)
            self.assertTrue((folder / "spectrum.png").is_file())
            text = (folder / "spectrum.csv").read_text(encoding="utf-8")
            self.assertIn("Fp1", text)
            self.assertIn("Eyes closed", text)


class _Frame:
    def __init__(self, columns: dict):
        self._columns = columns

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def __contains__(self, key: object) -> bool:
        return key in self._columns

    def __getitem__(self, key: str):
        return self._columns[key]


def _frame(columns: dict) -> _Frame:
    return _Frame(columns)


def _pack_mark(raw: bytearray, offset: int, onset_ms: int, label: str) -> None:
    encoded = label.encode("cp1252")
    struct.pack_into("<I", raw, offset, onset_ms)
    struct.pack_into("<H", raw, offset + 4, len(encoded))
    raw[offset + 6 : offset + 6 + len(encoded)] = encoded


if __name__ == "__main__":
    unittest.main()
