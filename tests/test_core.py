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
from bifooof.analysis.prepare import (
    apply_current_source_density, apply_reference, apply_vertex_reference,
    bipolar_derivations, global_scale_factor,
    CIRCUMFERENTIAL_BIPOLAR, LONGITUDINAL_BIPOLAR, TRANSVERSE_BIPOLAR,
    peak_at_width_floor, prepare_recording,
    sample_provenance, voltage_scale_note,
)
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
        self.assertEqual(clean_channel_name("LE"), "A1")
        self.assertEqual(clean_channel_name("RE"), "A2")
        self.assertEqual(clean_channel_name("T9"), "M1")
        self.assertEqual(clean_channel_name("T10"), "M2")
        self.assertEqual(clean_channel_name("EL"), "A1")
        self.assertEqual(clean_channel_name("ER"), "A2")

    def test_a_modern_name_wins_when_the_old_alias_is_also_present(self):
        legacy_first = finalize(np.array([[10.0, 0.0], [99.0, 99.0]]), ["T3", "T7"], 100.0, "edge.csv", "csv")
        self.assertEqual(legacy_first.ch_names, ["T7"])
        self.assertEqual(legacy_first.data[0, 0], 99.0)
        self.assertIn("Set aside T3 and kept T7", " ".join(legacy_first.notes))
        self.assertIn("T3", legacy_first.import_info["set_aside_channels"])

        modern_first = finalize(np.array([[99.0, 99.0], [10.0, 0.0]]), ["T7", "T3"], 100.0, "edge.csv", "csv")
        self.assertEqual(modern_first.data[0, 0], 99.0)

        alias_only = finalize(np.array([[5.0, 5.0]]), ["T3"], 100.0, "edge.csv", "csv")
        self.assertEqual(alias_only.ch_names, ["T7"])
        self.assertEqual(alias_only.data[0, 0], 5.0)
        self.assertNotIn("Set aside", " ".join(alias_only.notes))

        named = finalize(
            np.vstack([np.full(4, 10.0), np.full(4, 4.0), np.full(4, 8.0)]),
            ["O1", "LE", "T10"], 256.0, "ears.csv", "csv",
        )
        self.assertIn("A1", named.reference_channels)
        self.assertIn("M2", named.reference_channels)
        self.assertNotIn("LE", named.ch_names)
        self.assertNotIn("T10", named.ch_names)
        referenced = apply_reference(named, "left")
        self.assertEqual(referenced.reference_applied, "left")
        o1 = referenced.ch_names.index("O1")
        self.assertAlmostEqual(float(referenced.data[o1, 0]), 6.0)
        right = apply_reference(named, "right")
        self.assertEqual(right.reference_applied, "stored")
        self.assertNotIn("M2", right.notes[-1])
        self.assertAlmostEqual(float(right.data[right.ch_names.index("O1"), 0]), 10.0)

    def test_empty_edf_and_ragged_json_name_the_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.edf"
            empty.write_bytes(b"")
            with self.assertRaises(ValueError) as caught:
                load_recording(empty)
            self.assertIn("empty", str(caught.exception).lower())
            self.assertNotIn("invalid literal", str(caught.exception))

            ragged = root / "ragged.json"
            ragged.write_text('{"sfreq": 100, "data": [[1, 2], [3]]}', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_recording(ragged)
            self.assertIn("same number of samples", str(caught.exception))
            self.assertNotIn("inhomogeneous", str(caught.exception))

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

    def test_a_second_eyes_closed_mark_does_not_cut_the_first(self):
        recording = Recording(
            data=np.zeros((1, int(12 * 256))), ch_names=["O1"], sfreq=256.0,
            source_path="marked.csv", source_format="csv",
            marks=[Mark(0.0, "EC"), Mark(5.0, "still EC"), Mark(10.0, "EO")],
        )
        sides = split_marked_conditions(recording)
        self.assertAlmostEqual(sides["ec"].duration_s, 10.0, delta=0.02)
        self.assertAlmostEqual(sides["eo"].duration_s, 2.0, delta=0.02)
        self.assertNotIn("left out", sides["ec"].notes[-1])

    def test_a_separate_eyes_closed_span_is_named_when_it_is_left_out(self):
        recording = Recording(
            data=np.zeros((1, int(12 * 256))), ch_names=["O1"], sfreq=256.0,
            source_path="marked.csv", source_format="csv",
            marks=[Mark(0.0, "EC", 2.0), Mark(8.0, "EC", 3.0)],
        )
        closed = split_marked_conditions(recording)["ec"]
        self.assertAlmostEqual(closed.duration_s, 3.0, delta=0.02)
        self.assertIn("left out", closed.notes[-1])
        self.assertIn("2.0 s at 0.0 s", closed.notes[-1])

    def test_a_written_duration_stops_at_the_next_condition(self):
        recording = Recording(
            data=np.zeros((1, int(12 * 256))), ch_names=["O1"], sfreq=256.0,
            source_path="marked.csv", source_format="csv",
            marks=[Mark(0.0, "EC", 10.0), Mark(6.0, "EO", 4.0)],
        )
        sides = split_marked_conditions(recording)
        self.assertAlmostEqual(sides["ec"].duration_s, 6.0, delta=0.02)
        self.assertAlmostEqual(sides["eo"].duration_s, 4.0, delta=0.02)
        self.assertIn("cut at the next eyes open mark", sides["ec"].notes[-1])
        self.assertNotIn("cut at", sides["eo"].notes[-1])

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

    def test_welch_segment_stays_at_the_requested_length(self):
        sfreq = 100.0
        times = np.arange(int(60 * sfreq)) / sfreq
        data = (20.0 * np.sin(2 * np.pi * 10.0 * times))[None, :]
        _freqs, _power, nperseg = welch_spectra(data, sfreq, segment_s=2.0)
        self.assertEqual(nperseg, 200)

    def test_a_short_sine_is_not_called_flat(self):
        samples = 20.0 * np.sin(np.linspace(0, 6, 15))
        recording = Recording(samples[None, :], ["O1"], 128.0, "edge", "synthetic")
        with self.assertRaises(ValueError) as caught:
            fit_spectra(recording)
        message = str(caught.exception)
        self.assertIn("16", message)
        self.assertNotIn("flat", message.lower())

    def test_a_low_sampling_rate_says_the_range_was_brought_down(self):
        sfreq = 64.0
        times = np.arange(int(8 * sfreq)) / sfreq
        data = (20.0 * np.sin(2 * np.pi * 10.0 * times))[None, :]
        recording = Recording(data, ["O1"], sfreq, "edge", "synthetic")
        fit = fit_spectra(recording, FitSettings(fmin=1, fmax=40, segment_s=2))
        self.assertLess(fit.fmax, 32.0)
        self.assertIn("31.5", fit.notes)
        self.assertIn("brought down", fit.notes)
        self.assertEqual(fit.welch_nperseg, 128)

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

    def test_a_large_offset_does_not_reject_the_eeg(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        signal = 10000.0 + 20.0 * np.sin(2 * np.pi * 10.0 * times)
        signal[int(3 * sfreq):int(4 * sfreq)] += 5000.0
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="offset.csv", source_format="csv",
        )
        prepared = prepare_recording(recording, FitSettings())
        self.assertTrue(np.isfinite(prepared.data[0, : int(2 * sfreq)]).all())
        self.assertTrue(np.isnan(prepared.data[0, int(3 * sfreq):int(4 * sfreq)]).all())
        self.assertTrue(any("500" in note for note in prepared.notes))

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
        self.assertTrue(np.isnan(prepared.data[1, 4]))
        self.assertAlmostEqual(float(prepared.data[0, 0]), 0.0)
        self.assertAlmostEqual(float(prepared.data[1, 0]), 0.0)

    def test_a_rejected_sample_stays_out_of_the_reference(self):
        values = np.full(8, 10.0)
        spike = values.copy()
        spike[3] = 1000.0
        valid = np.ones((3, 8), dtype=bool)
        valid[0, 3] = False
        recording = Recording(
            np.vstack([spike, values, values]), ["Fp1", "Cz", "O1"], 256.0,
            "rejected.csv", "csv", valid=valid,
        )
        averaged = apply_reference(recording, "average")
        self.assertTrue(np.isnan(averaged.data[:, 3]).all())
        self.assertAlmostEqual(float(averaged.data[1, 0]), 0.0)
        self.assertEqual(float(recording.data[0, 3]), 1000.0)
        self.assertTrue(bool(recording.valid[1, 3]))

        vertex_valid = np.ones((2, 8), dtype=bool)
        vertex_valid[1, 3] = False
        vertex = apply_reference(
            Recording(
                np.vstack([values, spike]), ["O1", "Cz"], 256.0, "vertex.csv", "csv",
                valid=vertex_valid,
            ),
            "vertex",
        )
        self.assertTrue(np.isnan(vertex.data[0, 3]))
        self.assertAlmostEqual(float(vertex.data[0, 0]), 0.0)

        neighbors = np.zeros((4, 8))
        neighbors[0] = values
        hole = np.ones((4, 8), dtype=bool)
        hole[1, 3] = False
        neighbors[1, 3] = 1000.0
        laplacian = apply_reference(
            Recording(
                neighbors, ["Fp1", "Fp2", "F7", "F3"], 256.0, "lap.csv", "csv", valid=hole,
            ),
            "hjorth",
        )
        self.assertTrue(np.isnan(laplacian.data[0, 3]))
        self.assertTrue(np.isfinite(laplacian.data[0, 0]))

        csd_valid = np.ones((4, 8), dtype=bool)
        csd_valid[0, 3] = False
        csd_data = np.zeros((4, 8))
        csd_data[0, 3] = 1.0e6
        density = apply_reference(
            Recording(
                csd_data, ["Fp1", "Fz", "Cz", "Pz"], 256.0, "csd.csv", "csv", valid=csd_valid,
            ),
            "csd",
        )
        self.assertTrue(np.isnan(density.data[:, 3]).all())
        self.assertEqual(float(csd_data[0, 3]), 1.0e6)

        scalp = np.full(8, 10.0)
        left = np.full(8, 4.0)
        right = np.full(8, 2.0)
        ear_valid = np.ones((3, 8), dtype=bool)
        ear_valid[1, 3] = False
        linked = apply_reference(
            Recording(
                np.vstack([scalp, left, right]), ["O1", "A1", "A2"], 256.0, "ears.csv", "csv",
                valid=ear_valid, reference_channels={"A1": left.copy(), "A2": right.copy()},
            ),
            "linked",
        )
        self.assertTrue(np.isnan(linked.data[0, 3]))
        self.assertAlmostEqual(float(linked.data[0, 0]), 7.0)
        self.assertEqual(float(left[3]), 4.0)

        blanked = np.ones((2, 8), dtype=bool)
        blanked[0, :] = False
        loud = np.full((2, 8), 10.0)
        loud[0, :] = 5000.0
        kept = apply_reference(
            Recording(loud, ["Fp1", "O1"], 256.0, "blanked.csv", "csv", valid=blanked),
            "average",
        )
        self.assertEqual(kept.reference_applied, "stored")
        self.assertAlmostEqual(float(kept.data[1, 0]), 10.0)
        self.assertTrue(np.isnan(kept.data[0]).all())

    def test_a_second_pass_keeps_the_reference_that_was_already_applied(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        tone = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        recording = Recording(
            np.vstack([tone, 0.5 * tone]), ["O1", "O2"], sfreq, "again.csv", "csv",
        )
        plain = prepare_recording(recording, FitSettings(reference="stored", amplitude_limit_uv=0))
        self.assertEqual(plain.import_info["reference_requested"], "stored")
        averaged = prepare_recording(recording, FitSettings(reference="average", amplitude_limit_uv=0))
        again = prepare_recording(averaged, FitSettings(reference="stored", amplitude_limit_uv=0))
        self.assertEqual(again.reference_applied, "average")
        self.assertEqual(again.import_info["reference_requested"], "average")
        self.assertEqual(sample_provenance(again)["reference_fallback"], "")
        failed = prepare_recording(
            Recording(tone[None, :], ["O1"], sfreq, "one.csv", "csv"),
            FitSettings(reference="linked", amplitude_limit_uv=0),
        )
        kept = prepare_recording(failed, FitSettings(reference="stored", amplitude_limit_uv=0))
        self.assertEqual(kept.import_info["reference_requested"], "linked")
        self.assertEqual(kept.reference_applied, "stored")
        self.assertIn("stored reference was kept", sample_provenance(kept)["reference_fallback"])
        fit = fit_spectra(recording, FitSettings(reference="average", epoch_s=8.0, amplitude_limit_uv=0))
        self.assertEqual(fit.provenance["reference_requested"], "average")
        self.assertEqual(fit.provenance["reference_applied"], "average")
        self.assertTrue(fit.epochs)
        for epoch in fit.epochs:
            self.assertEqual(epoch.provenance["reference_requested"], "average")
            self.assertEqual(epoch.provenance["reference_applied"], "average")
            self.assertEqual(epoch.provenance["reference_fallback"], "")

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

    def test_current_source_density_removes_a_shared_signal_and_names_its_limits(self):
        sfreq = 256.0
        times = np.arange(int(2.0 * sfreq)) / sfreq
        shared = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        names = ["Fp1", "Fz", "Cz", "Pz"]
        common = apply_reference(
            Recording(np.tile(shared, (4, 1)), names, sfreq, "common.csv", "csv"),
            "csd",
        )
        self.assertEqual(common.reference_applied, "csd")
        self.assertIn("µV/m²", common.notes[-1])
        self.assertLess(float(np.nanmax(np.abs(common.data))), 1e-4)
        focal_data = np.zeros((4, times.size))
        focal_data[2] = shared
        focal = apply_reference(Recording(focal_data, names, sfreq, "focal.csv", "csv"), "csd")
        self.assertGreater(float(np.nanmax(np.abs(focal.data[2]))), float(np.nanmax(np.abs(focal.data[0]))))
        shifted = apply_reference(
            Recording(focal_data + 1000.0, names, sfreq, "shift.csv", "csv"),
            "csd",
        )
        self.assertLess(float(np.nanmax(np.abs(focal.data - shifted.data))), 1e-4)

        short = apply_reference(
            Recording(np.tile(shared, (3, 1)), names[:3], sfreq, "short.csv", "csv"),
            "csd",
        )
        self.assertEqual(short.reference_applied, "stored")
        self.assertIn("at least 4", short.notes[-1])
        self.assertAlmostEqual(float(short.data[0, 0]), float(shared[0]))
        blank = apply_reference(
            Recording(np.full((4, times.size), np.nan), names, sfreq, "blank.csv", "csv"),
            "csd",
        )
        self.assertEqual(blank.reference_applied, "stored")
        self.assertIn("no finite samples", blank.notes[-1])
        self.assertTrue(np.isnan(blank.data).all())

        extra = np.vstack([focal_data, shared])
        mixed = apply_reference(
            Recording(extra, [*names, "X1"], sfreq, "extra.csv", "csv"),
            "csd",
        )
        self.assertTrue(np.isnan(mixed.data[-1]).all())
        self.assertIn("X1", mixed.notes[-1])
        self.assertIn("10-20", mixed.notes[-1])

        hole = focal_data.copy()
        hole[0, 3] = np.nan
        punched = apply_reference(Recording(hole, names, sfreq, "hole.csv", "csv"), "csd")
        self.assertTrue(np.isnan(punched.data[:, 3]).all())
        self.assertTrue(np.isfinite(punched.data[:, 0]).all())
        self.assertIn("missing", punched.notes[-1])
        self.assertFalse(bool(punched.valid[:, 3].any()))

    def test_a_different_channel_set_builds_the_laplacian_quickly(self):
        import time

        from bifooof.analysis.prepare import _laplacian_matrix

        _laplacian_matrix(("Fp1", "Fz", "Cz", "Pz"))
        names = (
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
        )
        started = time.perf_counter()
        matrix = _laplacian_matrix(names)
        self.assertEqual(matrix.shape, (19, 19))
        self.assertLess(time.perf_counter() - started, 0.2)
        started = time.perf_counter()
        other = _laplacian_matrix(names[:-1])
        self.assertEqual(other.shape, (18, 18))
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertGreater(float(np.max(np.abs(matrix))), 0.0)

    def test_global_scale_uses_the_average_and_the_geometric_mean(self):
        self.assertAlmostEqual(global_scale_factor(np.array([[1.0], [np.e], [np.e ** 2]])), np.e, places=6)
        partial = global_scale_factor(np.array([[1.0], [np.e], [-1.0]]))
        self.assertAlmostEqual(partial, float(np.exp(0.5)), places=6)
        self.assertIsNone(global_scale_factor(np.zeros((2, 3))))

        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        tone = np.sin(2 * np.pi * 10.0 * times)
        data = np.vstack([40.0 * tone, 10.0 * tone])
        recording = Recording(data, ["O1", "O2"], sfreq, "scale.csv", "csv")
        scaled = prepare_recording(recording, FitSettings(reference="global", amplitude_limit_uv=0))
        self.assertEqual(scaled.reference_applied, "global")
        self.assertIn("Global scale factor", scaled.notes[-1])
        self.assertAlmostEqual(float(np.nanmax(np.abs(scaled.data.sum(axis=0)))), 0.0, places=6)
        _freqs, power, _nperseg = welch_spectra(scaled.data, sfreq, 1.0, 40.0, 4.0)
        self.assertAlmostEqual(global_scale_factor(power), 1.0, places=4)
        self.assertEqual(sample_provenance(scaled)["reference_contributors"], ["O1", "O2"])

        same = prepare_recording(
            Recording(np.tile(20.0 * tone, (2, 1)), ["O1", "O2"], sfreq, "same.csv", "csv"),
            FitSettings(reference="global", amplitude_limit_uv=0),
        )
        self.assertEqual(same.reference_applied, "average")
        self.assertIn("no positive", same.notes[-1])

        alone = prepare_recording(
            Recording((20.0 * tone)[None, :], ["O1"], sfreq, "alone.csv", "csv"),
            FitSettings(reference="global", amplitude_limit_uv=0),
        )
        self.assertEqual(alone.reference_applied, "stored")
        self.assertIn("at least two", alone.notes[-1])
        self.assertAlmostEqual(float(alone.data[0, 0]), float(20.0 * tone[0]))

        brief = np.sin(2 * np.pi * 10.0 * np.arange(10) / sfreq)
        too_short = prepare_recording(
            Recording(np.vstack([brief, brief * 0.5]), ["O1", "O2"], sfreq, "brief.csv", "csv"),
            FitSettings(reference="global", segment_s=4.0, amplitude_limit_uv=0),
        )
        self.assertEqual(too_short.reference_applied, "average")
        self.assertIn("needs a spectrum", too_short.notes[-1])
        FitSettings(reference="csd").validate()
        FitSettings(reference="global").validate()
        FitSettings(reference="vertex").validate()
        FitSettings(reference="ipsilateral").validate()
        FitSettings(reference="contralateral").validate()
        FitSettings(reference="left").validate()
        FitSettings(reference="right").validate()
        FitSettings(reference="mastoid").validate()
        FitSettings(reference="rest").validate()

    def test_rest_uses_a_spherical_model_and_refuses_without_geometry(self):
        sfreq = 256.0
        times = np.arange(int(2.0 * sfreq)) / sfreq
        shared = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        names = ["Fp1", "F3", "C3", "P3", "O1", "F4", "C4", "O2"]
        common = apply_reference(
            Recording(np.tile(shared, (len(names), 1)), names, sfreq, "rest-common.csv", "csv"),
            "rest",
        )
        self.assertEqual(common.reference_applied, "rest")
        self.assertIn("template geometry/spherical model", common.notes[-1])
        self.assertIn("microvolts", common.notes[-1])
        self.assertLess(float(np.nanmax(np.abs(common.data))), 1e-6)
        provenance = sample_provenance(common)
        self.assertEqual(provenance["rest_model"], "template geometry/spherical model")
        self.assertEqual(provenance["reference_contributors"], names)
        self.assertEqual(len(provenance["rest_sphere_m"]), 4)
        focal = np.zeros((len(names), times.size))
        focal[2] = shared
        transformed = apply_reference(Recording(focal, names, sfreq, "rest-focal.csv", "csv"), "rest")
        self.assertGreater(
            float(np.nanmax(np.abs(transformed.data[2]))),
            float(np.nanmax(np.abs(transformed.data[0]))),
        )
        punched = focal.copy()
        punched[0, 3] = np.nan
        gapped = apply_reference(Recording(punched, names, sfreq, "rest-gap.csv", "csv"), "rest")
        self.assertTrue(np.isnan(gapped.data[:, 3]).all())
        self.assertTrue(np.isfinite(gapped.data[:, 0]).all())
        spike = np.zeros((len(names), 4))
        spike[0, 1] = 1.0e6
        mask = np.ones(spike.shape, dtype=bool)
        mask[0, 1] = False
        rejected = apply_reference(
            Recording(spike, names, sfreq, "rest-bad.csv", "csv", valid=mask), "rest",
        )
        self.assertTrue(np.isnan(rejected.data[:, 1]).all())
        self.assertTrue(np.isfinite(rejected.data[:, 0]).all())
        self.assertEqual(float(spike[0, 1]), 1.0e6)
        import mne

        info = mne.create_info(names, sfreq, ch_types="eeg")
        info.set_montage("standard_1020", match_case=False, on_missing="raise", verbose="error")
        raw = mne.io.RawArray(focal * 1e-6, info, verbose="error")
        sphere_model = mne.make_sphere_model("auto", "auto", info, verbose="error")
        source = mne.setup_volume_source_space(
            sphere=sphere_model, pos=15.0, exclude=30.0, sphere_units="m", verbose="error",
        )
        forward = mne.make_forward_solution(
            info, trans=None, src=source, bem=sphere_model, meg=False, eeg=True, verbose="error",
        )
        raw.set_eeg_reference("REST", forward=forward, verbose="error")
        np.testing.assert_allclose(transformed.data, raw.get_data() * 1e6, rtol=1e-5, atol=1e-5)
        unnamed = apply_reference(
            Recording(np.tile(shared, (4, 1)), ["E1", "E2", "E3", "E4"], sfreq, "rest-none.csv", "csv"),
            "rest",
        )
        self.assertEqual(unnamed.reference_applied, "stored")
        self.assertIn("10-20 layout", unnamed.notes[-1])
        self.assertIn("stored reference was kept", unnamed.notes[-1])
        self.assertNotIn("average", unnamed.notes[-1].lower())
        with self.assertRaises(ValueError):
            FitSettings(reference="nope").validate()

    def test_local_laplacian_keeps_voltage_units_and_its_neighbor_list(self):
        sfreq = 256.0
        times = np.arange(int(1.0 * sfreq)) / sfreq
        shared = 12.0 * np.sin(2 * np.pi * 10.0 * times)
        names = ["Fp1", "Fp2", "F7", "F3"]
        common = apply_reference(
            Recording(np.tile(shared, (len(names), 1)), names, sfreq, "lap.csv", "csv"),
            "hjorth",
        )
        self.assertEqual(common.reference_applied, "hjorth")
        self.assertIn("microvolts", common.notes[-1])
        self.assertIn("equal", common.notes[-1])
        self.assertLess(float(np.nanmax(np.abs(common.data[0]))), 1e-6)
        self.assertTrue(np.isnan(common.data[1]).all())
        self.assertEqual(common.import_info["hjorth_neighbors"]["Fp1"], ["Fp2", "F7", "F3"])
        self.assertEqual(common.import_info["hjorth_weights"], "equal, no distance scaling")
        self.assertEqual(sample_provenance(common)["reference_contributors"], ["Fp1"])
        focal = np.zeros((len(names), times.size))
        focal[0] = shared
        transformed = apply_reference(Recording(focal, names, sfreq, "lap-focal.csv", "csv"), "hjorth")
        np.testing.assert_allclose(transformed.data[0], shared, atol=1e-6)
        punched = np.tile(shared, (len(names), 1))
        punched[2, 5] = np.nan
        gapped = apply_reference(Recording(punched, names, sfreq, "lap-gap.csv", "csv"), "hjorth")
        self.assertTrue(np.isnan(gapped.data[0, 5]))
        self.assertTrue(np.isfinite(gapped.data[0, 0]))
        again = apply_reference(common, "hjorth")
        self.assertEqual(again.reference_applied, "hjorth")
        self.assertIn("already applied", again.notes[-1])

    def test_an_incomplete_average_blanks_the_sample_instead_of_keeping_it(self):
        values = np.array([[10.0, 10.0], [20.0, 20.0], [90.0, np.nan]], dtype=float)
        averaged = apply_reference(
            Recording(values, ["Fp1", "O1", "Cz"], 256.0, "mean.csv", "csv"),
            "average",
        )
        self.assertEqual(averaged.reference_applied, "average")
        self.assertAlmostEqual(float(averaged.data[0, 0]), -30.0)
        self.assertTrue(np.isnan(averaged.data[:, 1]).all())
        self.assertIn("only one channel remains", averaged.notes[-1])
        alone = apply_reference(
            Recording(np.full((1, 4), 10.0), ["O1"], 256.0, "one.csv", "csv"),
            "average",
        )
        self.assertEqual(alone.reference_applied, "stored")
        self.assertAlmostEqual(float(alone.data[0, 0]), 10.0)

    def test_a_blank_ear_sample_does_not_become_the_other_ear(self):
        scalp = np.full((3, 8), 10.0)
        left = np.full(8, 4.0)
        right = np.full(8, 2.0)
        right[3] = np.nan
        recording = Recording(
            scalp, ["O1", "O2", "Fz"], 256.0, "ears.csv", "csv",
            reference_channels={"A1": left, "A2": right},
        )
        linked = apply_reference(recording, "linked")
        self.assertAlmostEqual(float(linked.data[0, 0]), 7.0)
        self.assertTrue(np.isnan(linked.data[:, 3]).all())
        ipsilateral = apply_reference(recording, "ipsilateral")
        self.assertAlmostEqual(float(ipsilateral.data[0, 3]), 6.0)
        self.assertTrue(np.isnan(ipsilateral.data[1, 3]))
        self.assertTrue(np.isnan(ipsilateral.data[2, 3]))

    def test_a_gap_is_not_joined_before_the_spectrum(self):
        stepped = np.zeros((1, 512))
        stepped[0, 256:] = 1000.0
        joined, _nperseg_joined = welch_spectra(stepped, 256.0, 1.0, 40.0, 1.0)[1:]
        gapped = stepped.copy()
        gapped[0, 248:264] = np.nan
        _freqs, separated, _nperseg = welch_spectra(gapped, 256.0, 1.0, 40.0, 1.0)
        self.assertGreater(float(np.nanmax(joined)), 1.0)
        self.assertLess(float(np.nanmax(separated)), 1e-6)

    def test_unequal_runs_are_weighted_by_welch_windows(self):
        from scipy.signal import detrend, welch

        sfreq = 256.0
        nperseg = 256
        times_short = np.arange(nperseg) / sfreq
        times_long = np.arange(nperseg * 2) / sfreq
        short = np.sin(2 * np.pi * 10.0 * times_short)
        long = 3.0 * np.sin(2 * np.pi * 10.0 * times_long)
        signal = np.concatenate([short, np.full(32, np.nan), long])[None, :]
        freqs, power, used = welch_spectra(signal, sfreq, 1.0, 40.0, 1.0)
        self.assertEqual(used, nperseg)
        noverlap = nperseg // 2
        step = nperseg - noverlap
        weights = [(len(piece) - noverlap) // step for piece in (short, long)]
        self.assertEqual(weights, [1, 3])
        spectra = [
            welch(detrend(piece, type="linear"), fs=sfreq, nperseg=nperseg, noverlap=noverlap)[1]
            for piece in (short, long)
        ]
        expected = (weights[0] * spectra[0] + weights[1] * spectra[1]) / sum(weights)
        by_samples = (short.size * spectra[0] + long.size * spectra[1]) / (short.size + long.size)
        band = (freqs >= 1.0) & (freqs <= 40.0)
        source_band = (welch(short, fs=sfreq, nperseg=nperseg)[0] >= 1.0) & (
            welch(short, fs=sfreq, nperseg=nperseg)[0] <= 40.0
        )
        np.testing.assert_allclose(power[0], expected[source_band], rtol=1e-6, atol=1e-12)
        self.assertGreater(float(np.max(np.abs(power[0] - by_samples[source_band]))), 1e-3)
        self.assertTrue(band.any())

    def test_global_scale_removes_a_uniform_amplitude_difference(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        quiet = np.vstack([
            10.0 * np.sin(2 * np.pi * 10.0 * times),
            4.0 * np.sin(2 * np.pi * 6.0 * times),
        ])
        settings = FitSettings(reference="global", amplitude_limit_uv=0, fmin=1.0, fmax=40.0, segment_s=4.0)
        low = prepare_recording(Recording(quiet, ["O1", "O2"], sfreq, "quiet.csv", "csv"), settings)
        high = prepare_recording(Recording(quiet * 2.0, ["O1", "O2"], sfreq, "loud.csv", "csv"), settings)
        low_factor = low.import_info["global_scale_factor"]
        high_factor = high.import_info["global_scale_factor"]
        self.assertAlmostEqual(high_factor / low_factor, 4.0, places=4)
        _freqs, power_low, _nperseg = welch_spectra(low.data, sfreq, 1.0, 40.0, 4.0)
        _freqs, power_high, _nperseg = welch_spectra(high.data, sfreq, 1.0, 40.0, 4.0)
        np.testing.assert_allclose(power_high, power_low, rtol=1e-4, atol=1e-8)
        averaged = prepare_recording(
            Recording(quiet, ["O1", "O2"], sfreq, "avg.csv", "csv"),
            FitSettings(reference="average", amplitude_limit_uv=0),
        )
        _freqs, power_average, _nperseg = welch_spectra(averaged.data, sfreq, 1.0, 40.0, 4.0)
        np.testing.assert_allclose(power_average / low_factor, power_low, rtol=1e-4, atol=1e-8)
        finite = np.isfinite(power_average) & (power_average > 0) & np.isfinite(power_low) & (power_low > 0)
        shift = np.log10(power_average[finite]) - np.log10(power_low[finite])
        np.testing.assert_allclose(shift, np.log10(low_factor), rtol=1e-5, atol=1e-6)
        provenance = sample_provenance(low)
        self.assertEqual(provenance["reference_requested"], "global")
        self.assertEqual(provenance["reference_applied"], "global")
        self.assertEqual(provenance["reference_fallback"], "")
        self.assertIn("channel-by-frequency", provenance["global_scale_definition"])

    def test_a_failed_reference_keeps_the_request_beside_what_was_applied(self):
        prepared = prepare_recording(
            Recording(np.ones((1, 8)), ["O1"], 256.0, "missing.csv", "csv"),
            FitSettings(reference="linked", amplitude_limit_uv=0),
        )
        provenance = sample_provenance(prepared)
        self.assertEqual(provenance["reference_requested"], "linked")
        self.assertEqual(provenance["reference_applied"], "stored")
        self.assertIn("not in the file", provenance["reference_fallback"])

    def test_overlapping_pages_keep_one_current_source_density_operator(self):
        sfreq = 256.0
        count = int(4.0 * sfreq)
        times = np.arange(count) / sfreq
        names = ["Fp1", "F3", "Fz", "Cz", "Pz"]
        signal = np.zeros((len(names), count))
        signal[3] = 30.0 * np.sin(2 * np.pi * 10.0 * times)
        first, note, applied = apply_current_source_density(signal[:, :300], names, None, names)
        second, _note, _applied = apply_current_source_density(signal[:, 100:400], names, None, names)
        self.assertTrue(applied)
        self.assertIn("Sphere radius", note)
        self.assertIn("sparse", note)
        np.testing.assert_allclose(first[:, 100:], second[:, :200], atol=1e-6)
        punched = signal[:, :200].copy()
        punched[0] = np.nan
        held, held_note, held_applied = apply_current_source_density(punched, names, None, names)
        shrunk, _shrunk_note, shrunk_applied = apply_current_source_density(punched, names, None, None)
        self.assertFalse(held_applied)
        self.assertIn("do not share a finite sample", held_note)
        self.assertTrue(shrunk_applied)
        self.assertTrue(np.array_equal(held, punched, equal_nan=True))

    def test_vertex_ipsilateral_and_contralateral_references(self):
        o1 = np.full(8, 10.0)
        o2 = np.full(8, 10.0)
        fz = np.full(8, 10.0)
        cz = np.full(8, 4.0)
        data = np.vstack([o1, o2, fz, cz])
        names = ["O1", "O2", "Fz", "Cz"]
        ears = {"A1": np.full(8, 3.0), "A2": np.full(8, 7.0)}
        recording = Recording(data, names, 256.0, "ref.csv", "csv", reference_channels=ears)

        vertex = apply_reference(recording, "vertex")
        self.assertEqual(vertex.reference_applied, "vertex")
        self.assertTrue(np.isnan(vertex.data[names.index("Cz")]).all())
        self.assertAlmostEqual(float(vertex.data[0, 0]), 6.0)
        self.assertEqual(sample_provenance(vertex)["reference_contributors"], ["Cz"])
        shown, note, applied = apply_vertex_reference(data, names)
        self.assertTrue(applied)
        self.assertAlmostEqual(float(shown[names.index("Cz"), 0]), 0.0)
        self.assertIn("minus Cz", note)
        missing = apply_reference(
            Recording(data[:3], names[:3], 256.0, "no-cz.csv", "csv"), "vertex",
        )
        self.assertEqual(missing.reference_applied, "stored")
        self.assertIn("Cz was not in the file", missing.notes[-1])
        hole = data.copy()
        hole[names.index("Cz"), 2] = np.nan
        punched = apply_reference(Recording(hole, names, 256.0, "cz-hole.csv", "csv"), "vertex")
        self.assertTrue(np.isnan(punched.data[:, 2]).all())
        self.assertFalse(bool(punched.valid[:, 2].any()))
        alone = apply_reference(
            Recording(cz[None, :], ["Cz"], 256.0, "only-cz.csv", "csv"), "vertex",
        )
        self.assertIn("besides Cz", alone.notes[-1])

        ipsilateral = apply_reference(recording, "ipsilateral")
        self.assertEqual(ipsilateral.reference_applied, "ipsilateral")
        self.assertAlmostEqual(float(ipsilateral.data[names.index("O1"), 0]), 7.0)
        self.assertAlmostEqual(float(ipsilateral.data[names.index("O2"), 0]), 3.0)
        self.assertAlmostEqual(float(ipsilateral.data[names.index("Fz"), 0]), 5.0)
        self.assertEqual(sample_provenance(ipsilateral)["reference_contributors"], ["A1", "A2"])
        contralateral = apply_reference(recording, "contralateral")
        self.assertAlmostEqual(float(contralateral.data[names.index("O1"), 0]), 3.0)
        self.assertAlmostEqual(float(contralateral.data[names.index("O2"), 0]), 7.0)
        self.assertAlmostEqual(float(contralateral.data[names.index("Fz"), 0]), 5.0)
        absent = apply_reference(Recording(data, names, 256.0, "no-ears.csv", "csv"), "ipsilateral")
        self.assertEqual(absent.reference_applied, "stored")
        self.assertIn("not in the file", absent.notes[-1])
        extra = np.vstack([data, np.full(8, 9.0)])
        mixed = apply_reference(
            Recording(extra, [*names, "X1"], 256.0, "extra.csv", "csv", reference_channels=ears),
            "ipsilateral",
        )
        self.assertTrue(np.isnan(mixed.data[-1]).all())
        self.assertIn("X1", mixed.notes[-1])
        ear_hole = {"A1": np.full(8, 3.0), "A2": np.full(8, 7.0)}
        ear_hole["A1"][4] = np.nan
        gapped = apply_reference(
            Recording(data, names, 256.0, "ear-hole.csv", "csv", reference_channels=ear_hole),
            "ipsilateral", 0.0,
        )
        self.assertTrue(np.isnan(gapped.data[names.index("O1"), 4]))
        self.assertTrue(np.isfinite(gapped.data[names.index("O2"), 4]))
        self.assertIn("missing", gapped.notes[-1])

        derived = bipolar_derivations(
            np.vstack([np.full(4, 5.0), np.full(4, 2.0), np.full(4, 1.0)]),
            ["Fp1", "F7", "O2"], LONGITUDINAL_BIPOLAR,
        )
        self.assertIsNotNone(derived)
        rows, labels, _anchors = derived
        self.assertEqual(labels, ["Fp1−F7"])
        self.assertAlmostEqual(float(rows[0, 0]), 3.0)
        hidden = bipolar_derivations(
            np.vstack([np.full(4, 5.0), np.full(4, 2.0)]),
            ["Fp1", "F7"], LONGITUDINAL_BIPOLAR, ["Fp1"],
        )
        self.assertIsNone(hidden)
        self.assertIn(("Fp1", "F3"), LONGITUDINAL_BIPOLAR)
        self.assertIn(("O1", "O2"), TRANSVERSE_BIPOLAR)
        self.assertIn(("Fp1", "Fp2"), CIRCUMFERENTIAL_BIPOLAR)
        self.assertIn(("F7", "Fp1"), CIRCUMFERENTIAL_BIPOLAR)

    def test_one_ear_reference_and_legacy_site_names(self):
        o1 = np.full(8, 10.0)
        left = apply_reference(
            Recording(
                o1[None, :], ["O1"], 256.0, "left.csv", "csv",
                reference_channels={"A1": np.full(8, 4.0), "A2": np.full(8, np.nan)},
            ),
            "left",
        )
        self.assertEqual(left.reference_applied, "left")
        self.assertAlmostEqual(float(left.data[0, 0]), 6.0)
        self.assertIn("A1", left.notes[-1])
        self.assertEqual(sample_provenance(left)["reference_contributors"], ["A1"])
        blocked = apply_reference(
            Recording(o1[None, :], ["O1"], 256.0, "right.csv", "csv", reference_channels={"A1": np.full(8, 4.0)}),
            "right",
        )
        self.assertEqual(blocked.reference_applied, "stored")
        self.assertIn("not in the file", blocked.notes[-1])
        self.assertAlmostEqual(float(blocked.data[0, 0]), 10.0)

        times = np.arange(int(8.0 * 256.0)) / 256.0
        brain = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        fallback = apply_reference(
            Recording(
                brain[None, :], ["O1"], 256.0, "mastoid.csv", "csv",
                reference_channels={"A1": np.full(times.shape, 5000.0), "M1": brain + 4.0},
            ),
            "left",
            500.0,
        )
        self.assertEqual(fallback.reference_applied, "stored")
        self.assertIn("A1", fallback.notes[-1])
        self.assertNotIn("M1", fallback.notes[-1])
        self.assertAlmostEqual(float(fallback.data[0, 0]), float(brain[0]))

        in_data = apply_reference(
            Recording(
                np.vstack([np.full(8, 10.0), np.full(8, 4.0), np.full(8, 9.0)]),
                ["O1", "A1", "A2"], 256.0, "rows.csv", "csv",
            ),
            "left",
        )
        self.assertEqual(in_data.reference_applied, "left")
        self.assertAlmostEqual(float(in_data.data[0, 0]), 6.0)
        self.assertAlmostEqual(float(in_data.data[1, 0]), 0.0)
        self.assertEqual(sample_provenance(in_data)["reference_contributors"], ["A1"])
        linked = apply_reference(
            Recording(
                np.vstack([np.full(8, 10.0), np.full(8, 2.0), np.full(8, 6.0)]),
                ["O1", "A1", "A2"], 256.0, "linked-rows.csv", "csv",
            ),
            "linked",
        )
        self.assertEqual(linked.reference_applied, "linked")
        self.assertAlmostEqual(float(linked.data[0, 0]), 6.0)

        hole = np.full(8, 4.0)
        hole[3] = np.nan
        gapped = apply_reference(
            Recording(np.full(8, 10.0)[None, :], ["O1"], 256.0, "gap.csv", "csv", reference_channels={"A1": hole}),
            "left",
        )
        self.assertTrue(np.isnan(gapped.data[0, 3]))
        self.assertFalse(gapped.valid[0, 3])
        self.assertIn("missing", gapped.notes[-1])

        legacy = apply_reference(
            Recording(
                np.vstack([np.full(8, 10.0), np.full(8, 10.0)]),
                ["T3", "T4"], 256.0, "legacy.csv", "csv",
                reference_channels={"A1": np.full(8, 4.0), "A2": np.full(8, 7.0)},
            ),
            "ipsilateral",
        )
        self.assertAlmostEqual(float(legacy.data[0, 0]), 6.0)
        self.assertAlmostEqual(float(legacy.data[1, 0]), 3.0)
        derived = bipolar_derivations(
            np.vstack([np.full(4, 5.0), np.full(4, 2.0)]),
            ["T3", "T5"], LONGITUDINAL_BIPOLAR,
        )
        self.assertIsNotNone(derived)
        rows, labels, _anchors = derived
        self.assertEqual(labels, ["T3−T5"])
        self.assertAlmostEqual(float(rows[0, 0]), 3.0)

    def test_an_offset_on_the_ears_is_still_a_linked_reference(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        offset = 2000.0 + 20.0 * np.sin(2 * np.pi * 10.0 * times)
        ears = {"A1": offset.copy(), "A2": offset.copy()}
        ears["A1"][100] = 9000.0
        recording = Recording(
            data=offset[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="ears.csv", source_format="csv", reference_channels=ears,
        )
        linked = apply_reference(recording, "linked", 500.0)
        self.assertIn("A1 and A2", linked.notes[-1])
        self.assertNotIn("not in the file", linked.notes[-1])
        self.assertTrue(np.isnan(linked.data[0, 100]))
        finite = linked.data[0, np.isfinite(linked.data[0])]
        self.assertLess(float(np.max(np.abs(finite))), 1.0)

    def test_a_flat_ear_beyond_the_limit_is_named(self):
        sfreq = 256.0
        signal = np.full(int(8.0 * sfreq), 20.0)
        recording = Recording(
            data=signal[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="flat-ear.csv", source_format="csv",
            reference_channels={"A1": np.full(signal.shape, 5000.0), "A2": np.full(signal.shape, 5000.0)},
        )
        linked = apply_reference(recording, "linked", 500.0)
        self.assertIn("were in the file", linked.notes[-1])
        self.assertIn("trend", linked.notes[-1])
        self.assertNotIn("not in the file", linked.notes[-1])
        self.assertAlmostEqual(float(linked.data[0, 0]), 20.0)

    def test_linked_ears_name_only_the_pair_that_was_used(self):
        sfreq = 256.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        offset = 2000.0 + 20.0 * np.sin(2 * np.pi * 10.0 * times)
        both = {
            "A1": offset.copy(), "A2": offset.copy(),
            "M1": np.full(times.shape, 50.0), "M2": np.full(times.shape, 80.0),
        }
        recording = Recording(
            data=offset[None, :], ch_names=["O1"], sfreq=sfreq,
            source_path="both-ears.csv", source_format="csv", reference_channels=both,
        )
        linked = apply_reference(recording, "linked", 500.0)
        self.assertIn("A1 and A2", linked.notes[-1])
        self.assertEqual(sample_provenance(linked)["reference_contributors"], ["A1", "A2"])
        self.assertLess(float(np.nanmax(np.abs(linked.data))), 1.0)
        fallback_ears = {
            "A1": np.full(times.shape, 5000.0), "A2": np.full(times.shape, 5000.0),
            "M1": offset.copy(), "M2": offset.copy(),
        }
        fallback = apply_reference(
            Recording(
                data=offset[None, :], ch_names=["O1"], sfreq=sfreq,
                source_path="mastoid.csv", source_format="csv", reference_channels=fallback_ears,
            ),
            "linked",
            500.0,
        )
        self.assertIn("A1 and A2", fallback.notes[-1])
        self.assertIn("trend", fallback.notes[-1])
        self.assertEqual(fallback.reference_applied, "stored")
        self.assertNotIn("M1 and M2", fallback.notes[-1])
        mastoid = apply_reference(
            Recording(
                data=offset[None, :], ch_names=["O1"], sfreq=sfreq,
                source_path="mastoid.csv", source_format="csv", reference_channels=fallback_ears,
            ),
            "mastoid",
            500.0,
        )
        self.assertEqual(mastoid.reference_applied, "mastoid")
        self.assertIn("M1 and M2", mastoid.notes[-1])
        self.assertEqual(sample_provenance(mastoid)["reference_contributors"], ["M1", "M2"])
        self.assertLess(float(np.nanmax(np.abs(mastoid.data))), 1.0)

    def test_epoch_fits_keep_the_sampling_rate_note(self):
        sfreq = 64.0
        short_times = np.arange(int(6.0 * sfreq)) / sfreq
        short = Recording(
            (20.0 * np.sin(2 * np.pi * 10.0 * short_times))[None, :],
            ["O1"], sfreq, "epoch-short.csv", "csv",
        )
        short_fit = fit_spectra(short, FitSettings(fmin=1, fmax=40, segment_s=10, epoch_s=8))
        self.assertIn("brought down", short_fit.notes)
        self.assertIn("31.5", short_fit.notes)
        self.assertIn("Welch segment", short_fit.notes)
        self.assertIn("shorter than one", short_fit.notes)
        self.assertIn("8 s", short_fit.notes)
        fractional = fit_spectra(short, FitSettings(fmin=1, fmax=40, segment_s=2, epoch_s=8.5))
        self.assertIn("8.5 s", fractional.notes)
        self.assertNotIn("9 s epoch", fractional.notes)
        times = np.arange(int(20.0 * sfreq)) / sfreq
        recording = Recording(
            (20.0 * np.sin(2 * np.pi * 10.0 * times))[None, :],
            ["O1"], sfreq, "epoch.csv", "csv",
        )
        fit = fit_spectra(recording, FitSettings(fmin=1, fmax=40, segment_s=2, epoch_s=8))
        self.assertEqual(fit.epoch_count, 2)
        self.assertIn("brought down", fit.notes)
        self.assertIn("31.5", fit.notes)
        self.assertIn("last 4", fit.notes)
        self.assertEqual(fit.notes.count("brought down"), 1)

    def test_a_rejected_epoch_is_named_and_the_leftover_keeps_its_fraction(self):
        sfreq = 256.0
        times = np.arange(int(round(22.4 * sfreq))) / sfreq
        tone = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        leftover = fit_spectra(
            Recording(tone[None, :], ["O1"], sfreq, "tail.csv", "csv"),
            FitSettings(epoch_s=12, segment_s=2),
        )
        self.assertIn("last 10.4", leftover.notes)
        self.assertNotIn("last 10 s", leftover.notes)
        n = int(24 * sfreq)
        gap = 20.0 * np.sin(2 * np.pi * 10.0 * np.arange(n) / sfreq)
        gap[int(8 * sfreq):int(16 * sfreq)] = 5000.0
        fit = fit_spectra(
            Recording(gap[None, :], ["O1"], sfreq, "gap.csv", "csv"),
            FitSettings(epoch_s=8, segment_s=2, amplitude_limit_uv=500),
        )
        self.assertEqual(fit.epoch_count, 2)
        self.assertIn("8–16 s was left out", fit.notes)
        self.assertIn("No accepted samples", fit.notes)
        flat = Recording(np.full((1, n), 5000.0), ["O1"], sfreq, "flat.csv", "csv")
        with self.assertRaisesRegex(ValueError, "Amplitude above the limit"):
            fit_spectra(flat, FitSettings(epoch_s=8, segment_s=2, amplitude_limit_uv=500))

    def test_a_channel_left_out_of_one_epoch_is_named(self):
        sfreq = 256.0
        count = int(24 * sfreq)
        times = np.arange(count) / sfreq
        occipital = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        occipital = occipital.copy()
        occipital[int(8 * sfreq):int(16 * sfreq)] = 5000.0
        central = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        fit = fit_spectra(
            Recording(np.vstack([occipital, central]), ["O1", "Cz"], sfreq, "gap.csv", "csv"),
            FitSettings(epoch_s=8, segment_s=2, amplitude_limit_uv=500),
        )
        by_name = {item.name: item for item in fit.channels}
        self.assertEqual(fit.epoch_count, 3)
        self.assertEqual(by_name["O1"].n_epochs, 2)
        self.assertEqual(by_name["Cz"].n_epochs, 3)
        self.assertIn("O1 at 8–16 s was left out", fit.notes)
        self.assertIn("No accepted samples", fit.notes)
        self.assertNotIn("Cz at 8–16", fit.notes)

    def test_nearest_peak_stays_out_of_the_band_number(self):
        outside = ChannelFit(name="C3", index=0, usable=True, peaks=[(12.4, 0.2, 1.0), (6.5, 0.4, 1.0)])
        found = nearest_outside_peak(outside, (8.0, 12.0))
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found[0], 12.4)
        self.assertAlmostEqual(found[1], 0.4)
        inside = ChannelFit(name="O1", index=1, usable=True, peaks=[(10.0, 1.0, 1.0)])
        self.assertIsNone(nearest_outside_peak(inside, (8.0, 12.0)))
        self.assertTrue(peak_at_width_floor(0.5, 0.5))
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

    def test_a_nonfinite_band_change_stays_blank(self):
        from bifooof.analysis.compare import ChannelComparison, band_reactivity_percent
        from bifooof.analysis.findings import _power_clause

        self.assertIsNone(band_reactivity_percent(float("inf"), float("inf")))
        self.assertIsNone(band_reactivity_percent(1.0, float("nan")))
        self.assertIsNone(band_reactivity_percent(1.0, 0.0))
        blank = _power_clause(
            ChannelComparison(name="O1", band_power_reactivity_percent=float("nan")),
            "Eyes closed", "Eyes open",
        )
        self.assertEqual(blank, "")

    def test_an_even_number_of_exponent_changes_uses_the_middle_pair(self):
        from bifooof.analysis.compare import ChannelComparison
        from bifooof.analysis.findings import _slope_sentence

        rows = [
            ChannelComparison(name="O1", exponent_delta=-2.0),
            ChannelComparison(name="O2", exponent_delta=-1.0),
            ChannelComparison(name="Pz", exponent_delta=0.0),
            ChannelComparison(name="Cz", exponent_delta=5.0),
        ]
        text = _slope_sentence(rows[0], rows, "Eyes closed", "Eyes open")
        self.assertIn("median change is -0.50", text)

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

    def test_a_fractional_fit_window_is_not_rounded_to_a_whole_second(self):
        from bifooof.analysis.findings import compile_report, report_text
        from bifooof.analysis.specparam_fit import SpectrumFit

        item = ChannelFit(
            name="O1", index=0, usable=True, exponent=1.0, r_squared=0.9, peaks=[(10.0, 1.0, 1.0)],
        )
        fit = SpectrumFit(channels=[item], fmin=1.0, fmax=31.5, t_start=0.6, t_end=8.4)
        recording = Recording(
            data=np.zeros((1, int(8.4 * 256))), ch_names=["O1"], sfreq=256.0,
            source_path="rec.edf", source_format="edf",
        )
        text = report_text(compile_report({"ec": recording}, {"ec": fit}, {}, (8.0, 12.0), "O1"))
        self.assertIn("0.6–8.4 s", text)
        self.assertIn("31.5 Hz", text)
        self.assertNotIn("1–8 s", text)
        long = Recording(
            data=np.zeros((1, int(round(10.4 * 256)))), ch_names=["O1"], sfreq=256.0,
            source_path="rec.edf", source_format="edf",
        )
        long_text = report_text(compile_report({"ec": long}, {"ec": fit}, {}, (8.0, 12.0), "O1"))
        self.assertIn("10.4 s", long_text)
        self.assertNotIn(", 10 s,", long_text)

    def test_a_missing_ear_is_not_described_as_the_reference_that_was_used(self):
        from bifooof.analysis.cycles import CycleFit
        from bifooof.analysis.findings import _MARGINALIA, compile_report, report_text
        from bifooof.analysis.specparam_fit import SpectrumFit
        from bifooof.ui.findings_panel import _html

        item = ChannelFit(
            name="O1", index=0, usable=True, exponent=1.2, r_squared=0.9,
            peaks=[(10.0, 0.8, 1.2)],
        )
        closed = SpectrumFit(channels=[item], fmin=1.0, fmax=40.0, t_start=0.0, t_end=60.0)
        opened = SpectrumFit(channels=[item], fmin=1.0, fmax=40.0, t_start=0.0, t_end=60.0)
        note = "Left ear was not in the file. The stored reference was kept."
        closed.notes = note
        opened.notes = "Left out samples beyond ±500 µV. " + note
        closed.provenance = {"reference_applied": "stored"}
        opened.provenance = {"reference_applied": "stored"}
        recording = Recording(
            data=np.zeros((1, 8)), ch_names=["O1"], sfreq=256.0,
            source_path="closed.edf", source_format="edf",
        )
        other = Recording(
            data=np.zeros((1, 8)), ch_names=["O1"], sfreq=256.0,
            source_path="open.edf", source_format="edf",
        )
        cycles = {"ec": CycleFit(
            channel="O1", channel_index=0, band=(8.0, 12.0), sfreq=256.0,
            filtered=np.zeros(8), n_cycles=4, n_burst_cycles=0, n_burst_episodes=0,
            burst_duration_s=0.0, burst_fraction=0.0, period_ms=100.0, rise_decay=0.5,
            peak_trough=0.5, volt_amp=10.0, burst_period_ms=None, burst_rise_decay=None,
            burst_peak_trough=None, burst_volt_amp=None, notes=note,
        )}
        sections = compile_report(
            {"ec": recording, "eo": other},
            {"ec": closed, "eo": opened},
            cycles, (8.0, 12.0), "O1", FitSettings(reference="left"),
        )
        text = report_text(sections)
        self.assertIn("Reference: as stored in the file.", text)
        self.assertIn("Left ear was not in the file.", text)
        self.assertEqual(text.count("Left ear was not in the file."), 1)
        self.assertNotIn("minus the left ear", text)
        self.assertIn("Left out samples beyond ±500 µV.", text)
        self.assertEqual(sum(line in text for line in _MARGINALIA), 1)
        for line in text.splitlines():
            self.assertFalse("O1:" in line and "P7:" in line)
        page = _html(sections)
        self.assertIn("font-style: italic", page)
        self.assertIn("Cambria", page)
        applied = SpectrumFit(channels=[item], fmin=1.0, fmax=40.0)
        applied.provenance = {"reference_applied": "left", "reference_contributors": ["A1"]}
        applied_text = report_text(compile_report(
            {"ec": recording}, {"ec": applied}, {}, (8.0, 12.0), "O1", FitSettings(reference="left"),
        ))
        self.assertIn("Reference: each channel minus A1.", applied_text)

    def test_a_short_display_filter_says_why_and_a_short_stretch_stays_visible(self):
        from bifooof.analysis.display import display_filter

        sfreq = 256.0
        short = 20.0 * np.sin(2 * np.pi * 10.0 * np.arange(20) / sfreq)
        with self.assertRaisesRegex(ValueError, "shorter than the display filter"):
            display_filter(short[None, :], sfreq, "1–30")
        values = np.full((1, 300), np.nan)
        long = 20.0 * np.sin(2 * np.pi * 10.0 * np.arange(200) / sfreq)
        island = 20.0 * np.sin(2 * np.pi * 10.0 * np.arange(12) / sfreq)
        values[0, :200] = long
        values[0, 250:262] = island
        result, note = display_filter(values, sfreq, "1–30")
        self.assertIn("left unfiltered", note)
        self.assertTrue(np.allclose(result[0, 250:262], island))
        self.assertFalse(np.allclose(result[0, :200], long))

    def test_a_band_range_may_name_its_unit(self):
        from bifooof.batch import resolve_band

        self.assertEqual(resolve_band("8-12 Hz"), (8.0, 12.0))
        self.assertEqual(resolve_band("8-12Hz"), (8.0, 12.0))
        self.assertEqual(resolve_band("8–12"), (8.0, 12.0))
        with self.assertRaises(ValueError) as caught:
            resolve_band("8-")
        self.assertIn("Unknown band", str(caught.exception))


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

    def test_the_reference_menu_separates_ears_mastoids_and_rest(self):
        from bifooof.ui.fit_bar import FitBar
        bar = FitBar()
        self.addCleanup(bar.close)
        self.assertGreaterEqual(bar.reference.findData("linked"), 0)
        self.assertGreater(bar.reference.findData("mastoid"), bar.reference.findData("linked"))
        self.assertGreaterEqual(bar.reference.findData("rest"), 0)
        self.assertGreaterEqual(bar.reference.findData("hjorth"), 0)
        bar.set_settings(FitSettings(reference="rest"))
        self.assertEqual(bar.settings().reference, "rest")
        bar.set_settings(FitSettings(reference="mastoid"))
        self.assertEqual(bar.settings().reference, "mastoid")
        for index in range(bar.reference.count()):
            bar.reference.setCurrentIndex(index)
            FitSettings(reference=bar.settings().reference).validate()

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
            self.assertFalse(any(text.get_text() == "EEG Paradox Solutions" for text in fig.texts))
        window.close()

    def test_one_sample_trace_has_a_width(self):
        from bifooof.ui.trace_panel import TracePanel
        panel = TracePanel()
        panel.set_recording(Recording(np.array([[12.0]]), ["O1"], 128.0, "edge", "synthetic"), 0)
        self.app.processEvents()
        axes = panel.canvas.fig.axes
        self.assertTrue(axes)
        left, right = axes[0].get_xlim()
        self.assertGreater(right - left, 0.0)
        panel.close()

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
        self.assertFalse(any(
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
            self.assertFalse(any(text.get_text() == "EEG Paradox Solutions" for text in fig.texts))
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
