"""Behavioral coverage for saved reviews and the expanded EEG workflow."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication

from bifooof.analysis.settings import FitSettings
from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit, fit_spectra
from bifooof.analysis.quality import QualitySpan
from bifooof.analysis.insights import quality_rows, comparison_checks, epoch_rows, stability_summary, alpha_candidates, regional_rows, asymmetry
from bifooof.events import event_table
from bifooof.export import write_csv_stream
from bifooof.io.load import load_recording
from bifooof.io.marks import split_marked_conditions
from bifooof.io.recording import Recording, Mark
from bifooof.jobs import display_envelope, JobToken, current_job, Cancelled, ResultCache, checkpoint
from bifooof.project import (ReviewState, atomic_json, make_project, read_project, load_project_sources,
    load_verified_source, profile, read_profile, fingerprint, verify_source, settings_from)
from bifooof.report import report_html


def source_file(folder, name="recording.json", seconds=32):
    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 128
    times = np.arange(seconds * rate) / rate
    rng = np.random.default_rng(23)
    data = np.vstack([20 * np.sin(2*np.pi*(10 + index*.1)*times) + rng.normal(0, 2, times.size) for index in range(2)])
    atomic_json(path, {"sfreq": rate, "unit": "uV", "channel_names": ["O1", "O2"], "reference": "declared test reference", "data": data.tolist(),
        "marks": [{"onset_s": 0, "label": "EC"}, {"onset_s": seconds/2, "label": "EO"}]}, backup=False)
    return path


def example_fit():
    frequencies = np.arange(1., 40.25, .25)
    return SpectrumFit([ChannelFit(name, index, True, exponent=1 + index, r_squared=.95, peaks=[(11., 1., 2.)],
        freqs=frequencies, spectrum=np.zeros(frequencies.size)) for index, name in enumerate(["O1", "O2"])], 1, 40, settings=FitSettings())


class ProjectTests(unittest.TestCase):
    def test_eeglab_declared_companion_and_invalid_view_controls(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as folder:
            companion = Path(folder)/"different-name.fdt"
            companion.write_bytes(b"sample bytes")
            header = Path(folder)/"recording.set"
            savemat(header, {"EEG": {"data": companion.name}})
            self.assertEqual({entry["role"] for entry in fingerprint(header)}, {"primary", "fdt"})
            source = load_verified_source(source_file(folder))
            project = make_project({"ec": source}, FitSettings(), FitSettings(), ReviewState(), {"trace": {"gain": "bad value"}})
            path = Path(folder)/"bad-project.json"
            atomic_json(path, project)
            with self.assertRaisesRegex(ValueError, "numeric control"):
                read_project(path)

    def test_source_crop_review_staged_state_and_history_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source_file(folder)
            source = load_verified_source(path)
            sides = split_marked_conditions(source)
            review = ReviewState()
            review.change(notes="Review note", bookmarks=[{"condition": "eo", "channel": "O1", "time_s": 2., "duration_s": 1., "label": "test"}],
                exclusions={"eo": [asdict(QualitySpan("O1", 2, 4, "user", "confirmed", "test"))]})
            project = make_project(sides, FitSettings(), FitSettings(t_start=1, t_length=5), review, {"channel": "O2"})
            destination = Path(folder) / "review.bifooof.json"
            atomic_json(destination, project)
            restored = read_project(destination)
            sources, continuous = load_project_sources(restored)
            self.assertIsNotNone(continuous)
            self.assertEqual(sources["eo"].source_start_s, 16.)
            state = ReviewState()
            state.restore(restored["review"])
            state.restore_history(restored["review_history"])
            masked = state.apply(sources["eo"], "eo")
            self.assertFalse(masked.mask(0)[2*128:4*128].any())
            self.assertTrue(masked.mask(1).all())
            self.assertEqual(restored["staged"]["t_start"], 1)
            state.undo()
            self.assertEqual(state.value["notes"], "")
            state.redo()
            self.assertEqual(state.value["notes"], "Review note")

    def test_relink_same_content_but_reject_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source_file(folder)
            source = load_verified_source(path)
            project = make_project({"ec": source}, FitSettings(), FitSettings(), ReviewState())
            moved = path.with_name("moved.json")
            path.rename(moved)
            loaded, _continuous = load_project_sources(project, {"ec": str(moved)})
            self.assertEqual(loaded["ec"].source_path, str(moved))
            value = json.loads(moved.read_text())
            value["data"][0][0] += 10
            atomic_json(moved, value)
            with self.assertRaisesRegex(ValueError, "changed"):
                load_project_sources(project, {"ec": str(moved)})

    def test_csv_without_a_time_column_reopens_with_saved_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plain.csv"
            path.write_text("O1\n" + "\n".join(map(str, range(1280))), encoding="utf-8")
            source = load_verified_source(path, 128.)
            project = make_project({"ec": source}, FitSettings(), FitSettings(), ReviewState())
            restored, _continuous = load_project_sources(project)
            self.assertEqual(restored["ec"].sfreq, 128.)

    def test_companion_files_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            header = folder / "record.vhdr"
            header.write_text("DataFile=record.eeg\nMarkerFile=record.vmrk\n")
            (folder / "record.eeg").write_bytes(b"EEG")
            (folder / "record.vmrk").write_text("marks")
            expected = fingerprint(header)
            self.assertEqual(len(expected), 3)
            (folder / "record.vmrk").write_text("changed marks")
            with self.assertRaises(ValueError):
                verify_source(header, expected)

    def test_failed_project_publish_preserves_the_previous_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.json"
            atomic_json(path, {"value": "previous"})
            with patch("bifooof.project.os.replace", side_effect=PermissionError("locked")), self.assertRaises(PermissionError):
                atomic_json(path, {"value": "new"})
            self.assertEqual(json.loads(path.read_text())["value"], "previous")
            self.assertEqual(json.loads(Path(str(path)+".previous").read_text())["value"], "previous")

    def test_undo_of_overlapping_review_spans_preserves_source_bad_data(self):
        source = Recording(np.ones((1, 1280)), ["O1"], 128, "synthetic", "test", valid=np.ones((1, 1280), bool))
        source.valid[0, :128] = False
        state = ReviewState()
        first = asdict(QualitySpan("O1", 2, 4, "user", "confirmed", "first"))
        second = asdict(QualitySpan("O1", 3, 5, "user", "confirmed", "second"))
        state.change(exclusions={"ec": [first]})
        state.change(exclusions={"ec": [first, second]})
        state.undo()
        result = state.apply(source, "ec")
        self.assertFalse(result.mask(0)[:128].any())
        self.assertFalse(result.mask(0)[2*128:4*128].any())
        self.assertTrue(result.mask(0)[4*128:5*128].all())

    def test_bad_channel_choice_is_separate_from_source_mask(self):
        source = Recording(np.ones((2, 100)), ["O1", "O2"], 10, "test", "test")
        state = ReviewState()
        state.change(bad_channels={"ec": ["O2"]})
        result = state.apply(source, "ec")
        self.assertFalse(result.mask(1).any())
        self.assertTrue(source.mask(1).all())
        self.assertIn("O2", result.bad_channels)


class AnalysisFeatureTests(unittest.TestCase):
    def test_auxiliary_types_are_review_only_and_do_not_pull_eeg_reference(self):
        import mne
        from bifooof.io.load import _recording_from_raw, _channel_policy
        from bifooof.analysis.prepare import apply_reference, sample_provenance
        from bifooof.analysis.cycles import fit_cycles
        times = np.arange(2048) / 128
        data = np.vstack([20*np.sin(2*np.pi*10*times), 10*np.sin(2*np.pi*10*times), np.full(2048, 500), np.ones(2048)]) * 1e-6
        raw = mne.io.RawArray(data, mne.create_info(["O1", "O2", "EOG", "STI014"], 128, ["eeg", "eeg", "eog", "stim"]), verbose=False)
        token = _channel_policy.set("all")
        try:
            source = _recording_from_raw(raw, Path("example.fif"), "fif")
        finally:
            _channel_policy.reset(token)
        self.assertEqual(source.ch_names, ["O1", "O2", "Eog"])
        prepared = apply_reference(source, "average")
        np.testing.assert_allclose(prepared.data[0], (source.data[0] - source.data[1])/2, atol=1e-10)
        np.testing.assert_equal(prepared.data[2], source.data[2])
        self.assertEqual(sample_provenance(prepared)["reference_contributors"], ["O1", "O2"])
        fit = fit_spectra(source)
        self.assertFalse(next(row for row in fit.channels if row.name == "Eog").usable)
        with self.assertRaisesRegex(ValueError, "trace review only"):
            fit_cycles(source, 2, (8, 12))
        from bifooof.analysis.coherence import coherence_pair
        with self.assertRaisesRegex(ValueError, "trace review only"):
            coherence_pair(source, FitSettings(), 0, 2)

    def test_display_keeps_extrema_and_gap_boundaries(self):
        for seconds in (5, 10, 15, 20, 30):
            values = np.zeros(seconds * 1024)
            values[123], values[1234] = 250, -250
            values[2301:2317] = np.nan
            points = display_envelope(values)
            self.assertEqual(np.nanmax(values[points]), 250)
            self.assertEqual(np.nanmin(values[points]), -250)
            self.assertIn(2300, points)
            self.assertIn(2301, points)
            self.assertIn(2316, points)
            self.assertIn(2317, points)

    def test_display_filters_do_not_modify_data_or_join_gaps(self):
        from bifooof.analysis.display import display_filter
        values = np.sin(np.arange(1000)[None, :] / 10)
        values[:, 400:450] = np.nan
        before = values.copy()
        result = display_filter(values, 128, "1–30")
        np.testing.assert_equal(values, before)
        self.assertTrue(np.isnan(result[:, 400:450]).all())
        self.assertFalse(np.allclose(result[:, :400], values[:, :400]))

    def test_epoch_results_retain_times_and_support(self):
        with tempfile.TemporaryDirectory() as folder:
            source = load_recording(source_file(folder, seconds=24))
            fit = fit_spectra(source, FitSettings(epoch_s=8))
        self.assertEqual([(epoch.t_start, epoch.t_end) for epoch in fit.epochs], [(0, 8), (8, 16), (16, 24)])
        rows = epoch_rows(fit)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["accepted_s"] == 8 for row in rows))
        self.assertTrue(stability_summary(rows))

    def test_quality_and_comparison_show_different_effective_support(self):
        a, b = example_fit(), example_fit()
        a.t_end = b.t_end = 20
        a.provenance = {"reference_applied": "average", "reference_contributors": ["O1", "O2"],
            "channels": {"O1": {"accepted_duration_s": 20}}}
        b.provenance = {"reference_applied": "stored", "reference_contributors": [],
            "channels": {"O1": {"accepted_duration_s": 8}}}
        sources = {key: Recording(np.ones((2, 2560)), ["O1", "O2"], 128, key, "test") for key in ("ec", "eo")}
        rows = quality_rows(sources, {"ec": a, "eo": b})
        self.assertEqual(next(row for row in rows if row["condition"] == "eo" and row["channel"] == "O1")["excluded_percent"], 60)
        checks = comparison_checks(sources, {"ec": a, "eo": b})
        self.assertTrue(any(row["check"] == "Usable signal: O1" for row in checks))
        self.assertTrue(any(row["check"] == "reference applied" for row in checks))

    def test_regional_summaries_use_common_channels_and_keep_missing_members(self):
        a, b = example_fit(), example_fit()
        b.channels[1].usable = False
        rows = regional_rows({"ec": a, "eo": b}, {"Posterior": ["O1", "O2"]}, (8, 12))
        self.assertTrue(all(row["channels"] == "O1" and row["missing"] == "O2" for row in rows))
        self.assertTrue(all(row["mean_band_psd_uv2_hz"] == 1 for row in rows))

    def test_asymmetry_formula_and_missing_data(self):
        rows = [{"condition": "ec", "region": "Left", "mean_band_psd_uv2_hz": 100},
                {"condition": "ec", "region": "Right", "mean_band_psd_uv2_hz": 10}]
        self.assertAlmostEqual(asymmetry(rows, "Left", "Right")[0]["log10_power_ratio"], 1)
        self.assertIsNone(asymmetry(rows, "Left", "Missing")[0]["log10_power_ratio"])

    def test_alpha_candidates_require_a_supported_actual_peak(self):
        fit = example_fit()
        self.assertEqual(alpha_candidates(fit, ["O1"])[0]["peak_hz"], 11)
        fit.channels[0].peaks = []
        self.assertFalse(alpha_candidates(fit, ["O1"]))

    def test_transition_directions_and_explicit_informational_mapping(self):
        source = Recording(np.ones((1, 100)), ["O1"], 1, "test", "test", marks=[Mark(0, "EC"), Mark(20, "EO"), Mark(40, "EC"), Mark(60, "EO")])
        rows = event_table(source)
        self.assertEqual([row["direction"] for row in rows], ["", "ec_to_eo", "eo_to_ec", "ec_to_eo"])
        rows[1]["state"] = ""
        self.assertEqual(event_table(source, rows)[1]["state"], "")
        rows[0]["onset"] = -1
        self.assertEqual(event_table(source, rows)[0]["onset"], -1)

    def test_reactivity_keeps_opposite_transition_groups_and_support_counts(self):
        from bifooof.ui.time_panel import _reactivity
        times = np.arange(4096) / 128
        source = Recording(np.array([20*np.sin(2*np.pi*10*times)]), ["O1"], 128, "test", "test",
            marks=[Mark(0, "EC"), Mark(8, "EO"), Mark(16, "EC"), Mark(24, "EO")])
        options = {"continuous": source, "band": (8, 12), "pre_s": 1, "post_s": 2}
        opened = _reactivity(source, FitSettings(), 0, {**options, "event_direction": "ec_to_eo"})
        closed = _reactivity(source, FitSettings(), 0, {**options, "event_direction": "eo_to_ec"})
        self.assertEqual(opened["events"], [8, 24])
        self.assertEqual(closed["events"], [16])
        self.assertEqual(int(opened["supports_change"].sum()), 2)
        self.assertIn("2/2 events", opened["note"])


class WorkerAndExportTests(unittest.TestCase):
    def test_batch_retry_preserves_earlier_successes_and_aggregate_rows(self):
        import csv
        from bifooof.batch_queue import run_queue
        with tempfile.TemporaryDirectory() as folder:
            first, second = source_file(folder), source_file(folder, "second.json")
            items = [{"index": index, "path": str(path), "condition": "ec"} for index, path in enumerate([first, second])]
            destination = Path(folder)/"output"
            def fake_run(path, output, **kwargs):
                output.mkdir()
                (output/"spectrum.csv").write_text("channel,exponent\nO1,1.5\n", encoding="utf-8")
            def first_attempt(path, output, **kwargs):
                if path == second:
                    raise ValueError("broken")
                fake_run(path, output, **kwargs)
            with patch("bifooof.batch_queue.run_file", side_effect=first_attempt):
                previous = run_queue(items, FitSettings(), destination)
            with patch("bifooof.batch_queue.run_file", side_effect=fake_run):
                retried = run_queue([items[1]], FitSettings(), destination)
            self.assertEqual([row["status"] for row in retried], ["Complete", "Complete"])
            self.assertEqual(retried[0]["output"], previous[0]["output"])
            with (destination/"batch-results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["batch_source"] for row in rows}, {str(first), str(second)})

    def test_review_mask_shares_samples_without_mutating_source(self):
        source = Recording(np.ones((2, 10000)), ["O1", "O2"], 128, "test", "test")
        state = ReviewState()
        state.change(exclusions={"ec": [asdict(QualitySpan("O1", 2, 4, "user", "confirmed", "review"))]})
        result = state.apply(source, "ec")
        self.assertTrue(np.shares_memory(source.data, result.data))
        self.assertTrue(np.isfinite(source.data).all())
        self.assertFalse(result.mask(0)[2*128:4*128].any())

    def test_batch_failure_is_isolated_and_same_basenames_have_distinct_outputs(self):
        from bifooof.batch_queue import run_queue
        with tempfile.TemporaryDirectory() as folder:
            first = source_file(Path(folder)/"a", seconds=12)
            second = source_file(Path(folder)/"b", seconds=12)
            broken = Path(folder)/"broken.json"
            broken.write_text("not JSON")
            results = run_queue([{"index": index, "path": str(path), "condition": "ec"} for index, path in enumerate([first, broken, second])], FitSettings(), Path(folder)/"output", ["O1", "O2"])
            self.assertEqual([row["status"] for row in results], ["Complete", "Failed", "Complete"])
            self.assertNotEqual(results[0]["output"], results[2]["output"])
            for result in (results[0], results[2]):
                manifest = json.loads((Path(result["output"])/"bifooof-manifest.json").read_text())
                self.assertTrue(manifest["complete"])
                import csv
                with (Path(result["output"])/"cycle_summary.csv").open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_cancelled_stream_does_not_replace_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "time.csv"
            path.write_text("previous complete result")
            token = JobToken()
            context = current_job.set(token)
            def rows():
                yield {"value": 1}
                token.stop.set()
                checkpoint()
                yield {"value": 2}
            try:
                with self.assertRaises(Cancelled):
                    write_csv_stream(path, rows())
            finally:
                current_job.reset(context)
            self.assertEqual(path.read_text(), "previous complete result")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_cache_is_bounded_and_recent_access_controls_eviction(self):
        cache = ResultCache(17000)
        cache["a"] = np.ones(1000)
        cache["b"] = np.ones(1000)
        cache.get("a")
        cache["c"] = np.ones(1000)
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        cache["too big"] = np.ones(10000)
        self.assertNotIn("too big", cache)
        self.assertLessEqual(cache.bytes_used, cache.max_bytes)

    def test_report_separates_notes_and_redacts_source_names(self):
        text = report_html({"alias": "Alias", "notes": "Reviewer <observation>", "findings": "private-name.json has a peak",
            "sources": [{"condition": "EC", "path": "/private/private-name.json"}], "settings": {}, "quality": [], "checks": []})
        self.assertIn("Reviewer &lt;observation&gt;", text)
        self.assertNotIn("private-name.json", text)
        self.assertIn("Calculated findings", text)


class DesktopFeatureTests(unittest.TestCase):
    def test_fractional_band_change_preserves_edges_and_recomputes_cycles(self):
        from bifooof.ui.cycle_panel import CyclePanel
        from bifooof.ui.fit_bar import FitBar
        cycles, bar = CyclePanel(), FitBar()
        self.addCleanup(cycles.close)
        self.addCleanup(bar.close)
        band = (8.0133, 11.9876)
        with patch.object(cycles, "_refit") as refit:
            cycles.set_band(*band, named=False)
            bar.set_band(*band, named=False)
            self.assertEqual(cycles.current_band(), band)
            self.assertEqual(bar.current_band(), band)
            refit.assert_called_once()
            cycles.set_band(*band, named=False)
            refit.assert_called_once()

    def test_trace_ruler_uses_samples_on_the_dragged_channel(self):
        from types import SimpleNamespace
        from bifooof.ui.trace_panel import TracePanel
        panel = TracePanel()
        self.addCleanup(panel.close)
        source = Recording(np.vstack([np.arange(1280), np.arange(1280)*2]), ["O1", "O2"], 128, "test", "test")
        panel.set_recording(source)
        panel._on_click(SimpleNamespace(inaxes=True, xdata=1., ydata=1.))
        panel._on_release(SimpleNamespace(xdata=2., ydata=-.8))
        self.assertIn("O1 displayed-sample Δamplitude +128.00 µV", panel.ruler.text())

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_profile_roundtrip_gui_and_cli_include_custom_settings(self):
        from bifooof.ui.fit_bar import FitBar
        from bifooof.cli import _parser, _settings
        chosen = FitSettings(t_start=3, t_length=None, segment_s=3, alpha_low=9, alpha_high=13, amp_fraction=.123, min_n_cycles=4, epoch_s=8)
        bar = FitBar()
        self.addCleanup(bar.close)
        bar.set_settings(chosen)
        self.assertEqual(bar.settings().key(), chosen.key())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            atomic_json(path, profile(chosen, name="Test profile"))
            args = _parser().parse_args(["run", "test.json", "--profile", str(path)])
            self.assertEqual(_settings(args).key(), bar.settings().key())

    def test_rapid_replacement_keeps_one_running_and_one_latest_request(self):
        import time
        from threading import Event
        from PySide6.QtWidgets import QWidget
        from bifooof.ui.workers import CallWorker, launch
        owner = QWidget()
        stop = Event()
        started, results = [], []
        def work(value):
            started.append(value)
            if value == 1:
                stop.wait(2)
            return value
        workers = []
        for generation in (1, 2, 3):
            worker = CallWorker(generation, work, (generation,))
            worker.succeeded.connect(lambda generation, value: results.append(value))
            launch(owner, worker)
            workers.append(worker)
        self.assertLessEqual(len(owner._live_workers), 2)
        stop.set()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and owner._live_workers:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(owner._live_workers)
        self.assertEqual(results, [3])
        self.assertNotIn(2, started)
        owner.close()

    def test_excluded_selected_cycle_channel_renders_an_empty_state(self):
        from bifooof.ui.cycle_panel import CyclePanel
        from bifooof.analysis.cycles import fit_cycles
        source = Recording(np.full((1, 2048), np.nan), ["O1"], 128, "test", "test")
        panel = CyclePanel()
        self.addCleanup(panel.close)
        panel.fit = fit_cycles(source, 0, (8, 12))
        panel._draw()
        self.assertTrue(any("No accepted" in text.get_text() for ax in panel.canvas.fig.axes for text in ax.texts))

    def test_project_restore_applies_review_and_keeps_staged_settings(self):
        from bifooof.ui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"BIFOOOF_STATE_DIR": folder}):
            source = load_verified_source(source_file(folder))
            sides = split_marked_conditions(source)
            state = ReviewState()
            state.change(notes="Restored review", bad_channels={"eo": ["O2"]}, events={source.source_path: [{"onset": 16, "trial_type": "Eyes open", "state": "eo"}]})
            project = make_project(sides, FitSettings(), FitSettings(t_start=2, t_length=5), state, {"active": "eo", "channel": "O1"})
            moved = replace(source, source_path=str(Path(folder)/"moved.json"))
            sides = split_marked_conditions(moved)
            with patch("bifooof.ui.spectrum_panel.launch"), patch("bifooof.ui.cycle_panel.launch"), patch("bifooof.ui.time_panel.launch"), patch("bifooof.ui.compare_panel.launch"):
                window = MainWindow()
                try:
                    window.workspace.restore_project(project, (sides, moved))
                    self.assertEqual(window.active, "eo")
                    self.assertEqual(window.workspace.review.notes.toPlainText(), "Restored review")
                    self.assertFalse(window.recording.mask(1).any())
                    self.assertEqual(window.fit_bar.settings().t_start, 2)
                    self.assertEqual(window.spectrum.settings.t_start, 0)
                    self.assertEqual(window.recording.source_start_s, 16)
                    self.assertEqual(window.time.review_events[moved.source_path][0]["onset"], 16)
                    self.assertNotIn(source.source_path, window.workspace.state.value["events"])
                    output = Path(folder)/"review-output"
                    output.mkdir()
                    window.workspace.write_review_outputs(output)
                    import csv
                    with (output/"events.tsv").open(encoding="utf-8", newline="") as stream:
                        events = list(csv.DictReader(stream, delimiter="\t"))
                    self.assertEqual(events[0]["condition"], "continuous")
                    self.assertEqual(events[0]["state"], "eo")
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
