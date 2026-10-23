"""Acceptance checks for the time-resolved measures.

These follow the cases named in docs/additional-graphs.md. They call the
analysis functions. They do not open a window.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from scipy.signal import sawtooth

from bifooof.analysis.coherence import coherence_pair
from bifooof.analysis.compare import compare_pair
from bifooof.analysis.cycles import BurstEpisode, fit_cycles
from bifooof.analysis.pac import phase_amplitude
from bifooof.analysis.quality import quality_spans
from bifooof.analysis.settings import FitSettings
from bifooof.analysis.shape import aligned_cycles, burst_stats, cycle_cloud
from bifooof.analysis.specparam_fit import ChannelFit, SpectrumFit, fit_spectra, log_residual
from bifooof.analysis.timefreq import (
    SlidingFit,
    align_to_events,
    channel_series,
    channel_spectrogram,
    exponent_spread,
    sliding_spectra,
    track_peaks,
    window_at,
)
from bifooof.io.recording import Mark, Recording
from bifooof.ui.time_panel import _reactivity


def _recording(data, names, sfreq, valid=None, marks=None) -> Recording:
    rows = np.atleast_2d(np.asarray(data, dtype=np.float64))
    return Recording(
        data=rows,
        ch_names=list(names),
        sfreq=float(sfreq),
        source_path="synthetic.dat",
        source_format="array",
        valid=None if valid is None else np.atleast_2d(np.asarray(valid, dtype=bool)),
        marks=list(marks or []),
    )


def _settings(**overrides) -> FitSettings:
    values = dict(segment_s=2.0, amplitude_limit_uv=0.0, fmin=1.0, fmax=40.0)
    values.update(overrides)
    return FitSettings(**values)


def _colored(n: int, sfreq: float, exponent: float, rng: np.random.Generator, scale: float = 15.0) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    freqs[0] = 1.0
    spectrum = spectrum / np.maximum(freqs, freqs[1]) ** (exponent / 2.0)
    wave = np.fft.irfft(spectrum, n)
    wave = wave - float(np.mean(wave))
    deviation = float(np.std(wave))
    if deviation > 0:
        wave = wave * (scale / deviation)
    return wave


def _peak_hz(spectrogram, column: int) -> float:
    power = spectrogram.power_db[:, column]
    finite = np.isfinite(power)
    self_index = int(np.argmax(np.where(finite, power, -np.inf)))
    return float(spectrogram.freqs_hz[self_index])


class SpectrogramTests(unittest.TestCase):
    def test_a_tone_change_and_a_gap_stay_on_the_clock(self):
        sfreq = 128.0
        seconds = 8.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        signal = np.where(
            times < 4.0,
            30.0 * np.sin(2 * np.pi * 10.0 * times),
            30.0 * np.sin(2 * np.pi * 20.0 * times),
        )
        valid = np.ones(signal.shape, dtype=bool)
        gap = (times >= 3.6) & (times < 4.4)
        signal = signal.copy()
        signal[gap] = np.nan
        valid[gap] = False
        recording = _recording(signal, ["Cz"], sfreq, valid=valid)
        result = channel_spectrogram(recording, _settings(), 0, window_s=1.0, overlap=0.5)

        early = [index for index, moment in enumerate(result.times_s) if moment < 3.2]
        late = [index for index, moment in enumerate(result.times_s) if moment > 4.8]
        self.assertTrue(early and late)
        self.assertLess(abs(_peak_hz(result, early[len(early) // 2]) - 10.0), result.df_hz + 0.01)
        self.assertLess(abs(_peak_hz(result, late[len(late) // 2]) - 20.0), result.df_hz + 0.01)
        blank = [
            moment for moment, column in zip(result.times_s, result.power_db.T)
            if not np.isfinite(column).any()
        ]
        self.assertTrue(blank)
        self.assertTrue(all(3.2 < moment < 4.8 for moment in blank))


class SlidingTests(unittest.TestCase):
    def test_a_peak_can_appear_without_a_large_exponent_step(self):
        sfreq = 128.0
        seconds = 16.0
        n = int(seconds * sfreq)
        rng = np.random.default_rng(4)
        background = _colored(n, sfreq, 1.0, rng, scale=12.0)
        times = np.arange(n) / sfreq
        tone = np.where(times >= 8.0, 25.0 * np.sin(2 * np.pi * 10.0 * times), 0.0)
        recording = _recording(background + tone, ["Cz"], sfreq)
        # A noise bump below this height is not the gated rhythm. The tone in the
        # second half is several times taller.
        sliding = sliding_spectra(
            recording, _settings(min_peak_height=0.8), window_s=8.0, overlap=0.0, channel_index=0,
        )
        series = channel_series(sliding, "Cz", (8.0, 12.0))
        self.assertEqual(series.peak_height.size, 2)
        self.assertFalse(np.isfinite(series.peak_height[0]))
        self.assertGreater(series.peak_height[1], 0.15)
        self.assertLess(abs(series.exponent[1] - series.exponent[0]), float(series.peak_height[1]))

    def test_tracks_follow_an_alpha_tone_and_ignore_a_beta_tone(self):
        windows = []
        centers = (9.0, 9.7, 10.4, 11.0)
        for index, center in enumerate(centers):
            item = ChannelFit(
                name="Cz", index=0, usable=True, exponent=1.0, r_squared=0.9,
                peaks=[(center, 0.4 + 0.05 * index, 1.2), (20.0, 1.2, 1.5)],
            )
            windows.append(SpectrumFit(
                channels=[item], fmin=1.0, fmax=40.0,
                t_start=float(index * 4), t_end=float((index + 1) * 4),
            ))
        sliding = SlidingFit(windows=windows, window_s=4.0, overlap=0.0, artifact_fraction=[0.0] * 4)
        tracks = track_peaks(sliding, "Cz", (8.0, 12.0))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].centers), 4)
        self.assertLess(tracks[0].centers[0], 9.2)
        self.assertGreater(tracks[0].centers[-1], 10.8)
        self.assertTrue(all(center < 12.0 for center in tracks[0].centers))

    def test_a_missing_window_ends_the_track(self):
        def window(start, peaks):
            item = ChannelFit(name="Cz", index=0, usable=True, peaks=peaks, exponent=1.0, r_squared=0.9)
            return SpectrumFit(channels=[item], fmin=1.0, fmax=40.0, t_start=start, t_end=start + 4.0)

        sliding = SlidingFit(
            windows=[
                window(0.0, [(9.0, 0.4, 1.0)]),
                window(4.0, [(10.0, 0.4, 1.0)]),
                window(8.0, []),
                window(12.0, [(11.0, 0.4, 1.0)]),
            ],
            window_s=4.0, overlap=0.0, artifact_fraction=[0.0] * 4,
        )
        tracks = track_peaks(sliding, "Cz", (8.0, 12.0))
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0].centers), 2)
        self.assertEqual(len(tracks[1].centers), 1)

    def test_exponent_spread_needs_three_windows_and_follows_the_spectrum(self):
        short = SlidingFit(
            windows=[
                SpectrumFit(channels=[ChannelFit(name="Cz", index=0, usable=True, exponent=1.0)], fmin=1, fmax=40),
                SpectrumFit(channels=[ChannelFit(name="Cz", index=0, usable=True, exponent=1.1)], fmin=1, fmax=40),
            ],
            window_s=4.0, overlap=0.0, artifact_fraction=[0.0, 0.0],
        )
        _names, medians, spreads, counts = exponent_spread(short)
        self.assertEqual(counts, [2])
        self.assertIsNone(medians[0])
        self.assertIsNone(spreads[0])

        sfreq = 128.0
        seconds = 18.0
        n = int(seconds * sfreq)
        rng = np.random.default_rng(11)
        shared = _colored(n, sfreq, 1.2, rng, scale=18.0)
        other = _colored(n, sfreq, 1.2, np.random.default_rng(12), scale=18.0)
        white = _colored(n, sfreq, 0.0, np.random.default_rng(13), scale=18.0)
        recording = _recording(np.vstack([shared, other, white]), ["Cz", "Pz", "Fz"], sfreq)
        sliding = sliding_spectra(
            recording, _settings(fmin=2.0, fmax=30.0, max_n_peaks=3),
            window_s=6.0, overlap=0.0,
        )
        names, medians, spreads, counts = exponent_spread(sliding)
        by_name = {name: index for index, name in enumerate(names)}
        self.assertGreaterEqual(counts[by_name["Cz"]], 3)
        self.assertIsNotNone(medians[by_name["Cz"]])
        self.assertIsNotNone(medians[by_name["Pz"]])
        self.assertIsNotNone(medians[by_name["Fz"]])
        self.assertLess(abs(medians[by_name["Cz"]] - medians[by_name["Pz"]]), 0.45)
        pink_spread = max(spreads[by_name["Cz"]], spreads[by_name["Pz"]])
        contrast = abs(medians[by_name["Fz"]] - medians[by_name["Cz"]])
        self.assertGreater(contrast, pink_spread * 2.0)

    def test_a_click_uses_the_window_that_contains_that_time(self):
        from matplotlib.figure import Figure

        from bifooof.ui.time_draw import draw_trajectory

        def fit(start, end):
            freqs = np.linspace(2.0, 30.0, 40)
            item = ChannelFit(
                name="O1", index=0, usable=True, exponent=1.2, r_squared=0.9,
                peaks=[(10.0, 0.4, 1.5)],
                freqs=freqs,
                spectrum=np.full(freqs.shape, -0.4),
                model=np.full(freqs.shape, -0.6),
            )
            return SpectrumFit(channels=[item], fmin=2.0, fmax=30.0, t_start=start, t_end=end)

        sliding = SlidingFit(
            windows=[fit(0.0, 10.0), fit(5.0, 15.0), fit(20.0, 30.0)],
            window_s=10.0, overlap=0.5, artifact_fraction=[0.0, 0.0, 0.0],
        )
        self.assertEqual(window_at(sliding, 7.0).t_start, 0.0)
        self.assertEqual(window_at(sliding, 12.0).t_start, 5.0)
        self.assertIsNone(window_at(sliding, 17.0))
        chosen = window_at(sliding, 22.0)
        series = channel_series(sliding, "O1", (8.0, 12.0))
        figure = Figure()
        draw_trajectory(figure, series, [], "", (chosen, chosen.channels[0], ""), (22.0, 10.0))
        self.assertIn("log10 power", [ax.get_ylabel() for ax in figure.axes])
        titles = " ".join(ax.get_title(loc="left") for ax in figure.axes)
        self.assertIn("20", titles)
        self.assertIn("30", titles)


class BurstAndShapeTests(unittest.TestCase):
    def test_three_gated_bursts_keep_their_times(self):
        sfreq = 250.0
        frequency = 10.0
        cycle = 1.0 / frequency
        burst_s = 10 * cycle
        gap_s = 8 * cycle
        pieces = []
        starts = []
        clock = 0.0
        for _index in range(3):
            starts.append(clock)
            count = int(round(burst_s * sfreq))
            pieces.append(40.0 * np.sin(2 * np.pi * frequency * np.arange(count) / sfreq))
            clock += burst_s
            gap = int(round(gap_s * sfreq))
            pieces.append(np.zeros(gap))
            clock += gap_s
        signal = np.concatenate(pieces)
        settings = _settings(
            amp_fraction=0.2, amp_consistency=0.2, period_consistency=0.2,
            monotonicity=0.4, min_n_cycles=3,
        )
        fit = fit_cycles(_recording(signal, ["Cz"], sfreq), 0, (8.0, 12.0), settings)
        self.assertEqual(len(fit.episodes), 3)
        # bycycle never counts the first or last cycle of a run as a burst.
        for episode, start in zip(fit.episodes, starts):
            self.assertLess(abs(episode.start_s - (start + cycle)), cycle)
            self.assertLess(abs((episode.end_s - episode.start_s) - (burst_s - 2 * cycle)), cycle)
            self.assertLess(abs(episode.n_cycles - 8), 2)
        self.assertIn("sample_peak", fit.cycles[0])
        self.assertIn("sample_last_trough", fit.cycles[0])

    def test_known_durations_match_occupancy(self):
        from bifooof.analysis.cycles import CycleFit

        episodes = [
            BurstEpisode(0.0, 0.40, 4, 100.0, 12.0),
            BurstEpisode(1.0, 1.50, 5, 100.0, 12.0),
            BurstEpisode(3.0, 4.20, 12, 100.0, 12.0),
        ]
        fit = CycleFit(
            channel="Cz", channel_index=0, band=(8.0, 12.0), sfreq=250.0,
            filtered=np.zeros(8), n_cycles=21, n_burst_cycles=21, n_burst_episodes=3,
            burst_duration_s=2.1, burst_fraction=0.40, period_ms=100.0, rise_decay=0.5,
            peak_trough=0.5, volt_amp=12.0, burst_period_ms=100.0, burst_rise_decay=0.5,
            burst_peak_trough=0.5, burst_volt_amp=12.0, episodes=episodes,
        )
        stats = burst_stats(fit)
        counts, _edges = np.histogram(stats.durations_s, bins=[0.0, 0.45, 1.0, 2.0])
        self.assertEqual(list(counts), [1, 1, 1])
        self.assertAlmostEqual(stats.occupancy, fit.burst_fraction)
        self.assertAlmostEqual(stats.median_s, 0.5)

    def test_a_sine_is_symmetric_and_a_fast_rise_is_not(self):
        sfreq = 256.0
        seconds = 4.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        sine = 30.0 * np.sin(2 * np.pi * 10.0 * times)
        # scipy sawtooth width is the fraction of the period spent rising.
        # bycycle time_rdsym is that same rise fraction, so a width of 0.2 sits below 0.5.
        saw = 30.0 * sawtooth(2 * np.pi * 10.0 * times, width=0.2)
        settings = _settings(monotonicity=0.3, amp_consistency=0.2, period_consistency=0.2, min_n_cycles=2)
        band = (2.0, 40.0)
        sine_fit = fit_cycles(_recording(sine, ["Cz"], sfreq), 0, band, settings)
        saw_fit = fit_cycles(_recording(saw, ["Cz"], sfreq), 0, band, settings)
        self.assertAlmostEqual(sine_fit.rise_decay, 0.5, delta=0.12)
        self.assertAlmostEqual(sine_fit.peak_trough, 0.5, delta=0.12)
        self.assertLess(saw_fit.rise_decay, 0.42)
        self.assertLess(saw_fit.rise_decay, sine_fit.rise_decay)

        sine_peak = _median_peak(*aligned_cycles(sine_fit, align="trough")[:2])
        saw_peak = _median_peak(*aligned_cycles(saw_fit, align="trough")[:2])
        self.assertLess(saw_peak, sine_peak)

    def test_one_frequency_stays_narrow_and_a_sweep_does_not(self):
        sfreq = 250.0
        seconds = 6.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        amplitude = 15.0 + 25.0 * (times / seconds)
        steady = amplitude * np.sin(2 * np.pi * 10.0 * times)
        sweep_hz = 8.0 + 4.0 * (times / seconds)
        sweep = 30.0 * np.sin(2 * np.pi * np.cumsum(sweep_hz) / sfreq)
        settings = _settings(amp_consistency=0.0, period_consistency=0.0, monotonicity=0.2, min_n_cycles=2)
        steady_fit = fit_cycles(_recording(steady, ["Cz"], sfreq), 0, (8.0, 12.0), settings)
        sweep_fit = fit_cycles(_recording(sweep, ["Cz"], sfreq), 0, (6.0, 16.0), settings)
        steady_hz, steady_uv, _bursts = cycle_cloud(steady_fit)
        sweep_freq, _sweep_uv, _ignored = cycle_cloud(sweep_fit)
        freq_iqr = float(np.subtract(*np.percentile(steady_hz, [75, 25])))
        amp_rel = float(np.subtract(*np.percentile(steady_uv, [75, 25])) / np.median(steady_uv))
        self.assertLess(freq_iqr / np.median(steady_hz), amp_rel)
        self.assertGreater(float(np.subtract(*np.percentile(sweep_freq, [75, 25]))), 1.0)


class ReactivityTests(unittest.TestCase):
    def test_a_mark_shows_the_step_and_split_files_do_not(self):
        sfreq = 128.0
        seconds = 24.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        mark = 10.0
        amplitude = np.where(times < mark, 40.0, 6.0)
        signal = amplitude * np.sin(2 * np.pi * 10.0 * times)
        recording = _recording(signal, ["O1"], sfreq, marks=[Mark(mark, "Eyes open")])
        settings = _settings(segment_s=1.0)
        around = _reactivity(recording, settings, 0, {
            "band": (8.0, 12.0),
            "channel": "O1",
            "continuous": recording,
        })
        self.assertEqual(around["kind"], "events")
        median = _finite_median(around["curves"])
        before = median[around["grid"] < -0.5]
        after = median[(around["grid"] >= 0.5) & (around["grid"] <= 5.0)]
        self.assertGreater(float(np.nanmedian(before)), float(np.nanmedian(after)) * 2.0)
        closed = _recording(signal[: int(mark * sfreq)], ["O1"], sfreq)
        opened = _recording(signal[int(mark * sfreq):], ["O1"], sfreq)
        separate = _reactivity(opened, settings, 0, {
            "band": (8.0, 12.0),
            "channel": "O1",
            "pair": {"ec": closed, "eo": opened},
        })
        self.assertEqual(separate["kind"], "separate")
        self.assertNotIn("curves", separate)
        self.assertIn("no transition", separate["note"])

    def test_a_gap_between_samples_does_not_become_the_step(self):
        times = np.concatenate([np.arange(0.0, 10.0, 1.0), np.arange(80.0, 90.0, 1.0)])
        grid, curves = align_to_events(times, np.ones(times.size), [5.0, 85.0], 4.0, 4.0)
        self.assertLess(float(grid[1] - grid[0]), 2.0)
        self.assertEqual(curves.shape[0], 2)
        self.assertTrue(np.isfinite(curves[0]).any())
        self.assertTrue(np.isfinite(curves[1]).any())


class CoherenceTests(unittest.TestCase):
    def test_a_shared_tone_peaks_and_independent_noise_does_not(self):
        sfreq = 256.0
        seconds = 8.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        tone = np.sin(2 * np.pi * 10.0 * times)
        shared = np.vstack([tone, np.sin(2 * np.pi * 10.0 * times + np.pi / 3.0)])
        rng = np.random.default_rng(3)
        noise = rng.normal(scale=1.0, size=shared.shape)
        settings = _settings(segment_s=1.0)
        coupled = coherence_pair(_recording(shared, ["C3", "C4"], sfreq), settings, 0, 1)
        alone = coherence_pair(_recording(noise, ["C3", "C4"], sfreq), settings, 0, 1)
        band = (coupled.freqs_hz >= 8.0) & (coupled.freqs_hz <= 12.0)
        peak = float(np.max(coupled.coherence[band]))
        noise_peak = float(np.max(alone.coherence[(alone.freqs_hz >= 8.0) & (alone.freqs_hz <= 12.0)]))
        at_ten = int(np.argmin(np.abs(coupled.freqs_hz - 10.0)))
        self.assertGreater(coupled.coherence[at_ten], 0.8)
        self.assertGreater(peak, noise_peak * 2.0)


class CouplingTests(unittest.TestCase):
    def test_locked_amplitude_beats_its_surrogates_and_an_unlocked_pair_does_not(self):
        sfreq = 250.0
        seconds = 24.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        # A pure sine stays locked after a time shift, because that shift only
        # moves which phase is preferred and the index ignores the preferred phase.
        # A wandering slow frequency makes the same shift break the lock.
        freq = 6.0 + 1.2 * np.sin(2 * np.pi * 0.2 * times)
        slow = np.sin(2 * np.pi * np.cumsum(freq) / sfreq)
        envelope = np.clip(slow, 0, None) ** 2
        locked = slow + (0.15 + envelope) * np.sin(2 * np.pi * 40.0 * times)
        settings = _settings()
        coupled = phase_amplitude(
            _recording(locked, ["Cz"], sfreq), settings, 0,
            (4.0, 9.0), (30.0, 50.0), n_surrogates=40, minimum_s=4.0, seed=1,
        )
        self.assertIsNotNone(coupled.mi)
        self.assertGreater(coupled.mi, coupled.surrogate_mean + 2.0 * coupled.surrogate_sd)
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            free = np.sin(2 * np.pi * 6.0 * times + rng.uniform(0, 6)) + np.sin(2 * np.pi * 40.0 * times)
            result = phase_amplitude(
                _recording(free, ["Cz"], sfreq), settings, 0,
                (4.0, 9.0), (30.0, 50.0), n_surrogates=20, minimum_s=4.0, seed=seed,
            )
            self.assertLess(result.mi, coupled.mi / 5.0)
            # Both sides are numerical zeros. A pure time shift does not create a peak.
            floor = max(float(result.surrogate_sd), 1e-4)
            self.assertLess(result.mi, result.surrogate_mean + 2.0 * floor)

    def test_a_harmonic_in_the_amplitude_band_is_not_computed(self):
        sfreq = 200.0
        times = np.arange(int(6.0 * sfreq)) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * times) + np.sin(2 * np.pi * 20.0 * times)
        result = phase_amplitude(
            _recording(signal, ["Cz"], sfreq), _settings(), 0,
            (8.0, 12.0), (15.0, 25.0), n_surrogates=4, minimum_s=4.0,
            nonsinusoidal=True,
        )
        self.assertIsNone(result.mi)
        self.assertIsNone(result.histogram)
        self.assertIn("not computed", result.note)
        self.assertIn("harmonic", result.note.lower())

    def test_a_harmonic_is_still_shown_when_the_cycles_are_not_marked_nonsinusoidal(self):
        sfreq = 200.0
        times = np.arange(int(6.0 * sfreq)) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * times) + 0.2 * np.random.default_rng(1).normal(size=times.size)
        result = phase_amplitude(
            _recording(signal, ["Cz"], sfreq), _settings(), 0,
            (8.0, 12.0), (15.0, 25.0), n_surrogates=4, minimum_s=4.0,
        )
        self.assertIsNotNone(result.mi)
        self.assertTrue(result.warnings)
        self.assertIn("still shown", result.warnings[0])

    def test_a_pure_sine_does_not_beat_a_time_shift(self):
        sfreq = 200.0
        times = np.arange(int(8.0 * sfreq)) / sfreq
        slow = np.sin(2 * np.pi * 6.0 * times)
        signal = slow + (0.2 + np.clip(slow, 0, None)) * np.sin(2 * np.pi * 40.0 * times)
        result = phase_amplitude(
            _recording(signal, ["Cz"], sfreq), _settings(fmax=55.0), 0,
            (4.0, 8.0), (30.0, 50.0), n_surrogates=20, minimum_s=4.0, seed=1,
        )
        self.assertIsNotNone(result.mi)
        self.assertLess(result.mi, result.surrogate_mean + 2.0 * max(result.surrogate_sd, 1e-4))
        self.assertIn("periodic envelope", result.note)

    def test_the_map_states_how_many_cells_were_tested(self):
        from matplotlib.figure import Figure

        from bifooof.ui.time_draw import draw_pac

        sfreq = 200.0
        times = np.arange(int(6.0 * sfreq)) / sfreq
        signal = np.sin(2 * np.pi * 6.0 * times) + np.sin(2 * np.pi * 40.0 * times)
        result = phase_amplitude(
            _recording(signal, ["Cz"], sfreq), _settings(fmax=55.0), 0,
            (4.0, 8.0), (30.0, 50.0), n_surrogates=4, minimum_s=4.0,
            comodulogram_grid=(np.array([6.0, 10.0]), np.array([30.0, 40.0])),
        )
        self.assertEqual(result.n_cells, 4)
        self.assertEqual(result.comodulogram.shape, (2, 2))
        self.assertIn("4 map cells", result.note)
        figure = Figure()
        draw_pac(figure, result)
        titles = " ".join(ax.get_title(loc="left") for ax in figure.axes)
        self.assertIn("4 cells, uncorrected", titles)


class QualityTests(unittest.TestCase):
    def test_clip_flat_and_a_step_are_marked_and_a_sine_is_not(self):
        sfreq = 256.0
        seconds = 8.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        signal = 20.0 * np.sin(2 * np.pi * 10.0 * times)
        signal = signal.copy()
        signal[(times >= 1.0) & (times < 2.2)] = 100.0
        signal[(times >= 3.0) & (times < 4.2)] = 0.0
        signal[(times >= 6.0) & (times < 6.05)] += 400.0
        spans = quality_spans(_recording(signal, ["Cz"], sfreq), limit_uv=0.0)
        self.assertTrue(_covers(spans, "clip", 1.0, 2.2))
        self.assertTrue(_covers(spans, "flat", 3.0, 4.2))
        self.assertTrue(_covers(spans, "step", 5.9, 6.2))

        clean = 25.0 * np.sin(2 * np.pi * 10.0 * times)
        detected = [span for span in quality_spans(_recording(clean, ["Cz"], sfreq), limit_uv=500.0) if span.level == "detected"]
        self.assertEqual(detected, [])


class ResidualTests(unittest.TestCase):
    def test_a_modeled_peak_leaves_a_smaller_residual_than_a_disallowed_peak(self):
        sfreq = 128.0
        seconds = 10.0
        times = np.arange(int(seconds * sfreq)) / sfreq
        rng = np.random.default_rng(8)
        signal = _colored(times.size, sfreq, 1.0, rng, scale=12.0) + 35.0 * np.sin(2 * np.pi * 10.0 * times)
        recording = _recording(signal, ["Cz"], sfreq)
        fitted = fit_spectra(recording, _settings())
        blocked = fit_spectra(recording, _settings(min_peak_height=8.0))
        good = next(item for item in fitted.channels if item.name == "Cz")
        missed = next(item for item in blocked.channels if item.name == "Cz")
        good_freq, good_residual = log_residual(good)
        missed_freq, missed_residual = log_residual(missed)
        good_rms = float(np.sqrt(np.mean(np.square(good_residual[np.isfinite(good_residual)]))))
        missed_rms = float(np.sqrt(np.mean(np.square(missed_residual[np.isfinite(missed_residual)]))))
        near = int(np.argmin(np.abs(missed_freq - 10.0)))
        good_near = int(np.argmin(np.abs(good_freq - 10.0)))
        self.assertGreater(abs(float(missed_residual[near])), abs(float(good_residual[good_near])))
        self.assertLess(good_rms, missed_rms)
        same = compare_pair(fitted, fitted, (8.0, 12.0))
        self.assertAlmostEqual(same[0].alpha_reactivity or 0.0, 0.0, places=6)
        self.assertAlmostEqual(same[0].exponent_delta or 0.0, 0.0, places=6)
        empty = SpectrumFit(
            channels=[ChannelFit(name="Cz", index=0, usable=True, peaks=[], exponent=1.0)],
            fmin=1.0, fmax=40.0,
        )
        self.assertIsNone(compare_pair(fitted, empty, (8.0, 12.0))[0].alpha_reactivity)


def _median_peak(grid: np.ndarray, waves: np.ndarray) -> float:
    covered = np.isfinite(waves).mean(axis=0) >= 0.5
    median = np.nanmedian(waves[:, covered], axis=0)
    return float(grid[covered][int(np.nanargmax(median))])


def _finite_median(curves: np.ndarray) -> np.ndarray:
    median = np.full(curves.shape[-1], np.nan)
    for index in range(curves.shape[-1]):
        column = curves[..., index]
        column = column[np.isfinite(column)]
        if column.size:
            median[index] = float(np.median(column))
    return median


def _covers(spans, kind: str, start: float, end: float) -> bool:
    return any(
        span.kind == kind and span.level == "detected" and span.start_s < end and span.end_s > start
        for span in spans
    )


if __name__ == "__main__":
    unittest.main()
