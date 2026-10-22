"""Loader, fit, and offscreen window checks."""

from __future__ import annotations

import os
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

from bifooof.analysis.compare import ChannelComparison, alpha_peak, compare_spectra
from bifooof.analysis.cycles import fit_cycles
from bifooof.analysis.settings import FitSettings, band_label, band_phrase
from bifooof.analysis.specparam_fit import ChannelFit, fit_spectra
from bifooof.export import comparison_rows
from bifooof.io.channels import clean_channel_name
from bifooof.io.condition import guess_condition
from bifooof.io.load import load_recording, needs_sfreq
from bifooof.io.recording import Recording
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
        low, high = window.fit_bar.current_band()
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 4.0)
        self.assertTrue(window.fit_bar.band.currentText().startswith("Delta"))
        for fig in (window.trace.canvas.fig, window.spectrum.canvas.fig, window.cycles.canvas.fig):
            self.assertTrue(any(text.get_text() == "EEG Paradox Solutions" for text in fig.texts))
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
            self.assertIn("band_low_hz", text)
            self.assertIn("band_high_hz", text)
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


if __name__ == "__main__":
    unittest.main()
