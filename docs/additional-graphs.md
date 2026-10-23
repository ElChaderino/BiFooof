# Additional graphs — implementation spec

The window's current behavior is [running.md](running.md). This file is the plan those views were built from. An item here is not in the program until `running.md` says so. When a planned piece is built, describe it there and remove it from that file's "Still open" list.

This is the build order for the time-resolved and multichannel views. Each view reads the recording, the fit settings, and the results that already exist. A new graph does not get its own filter, reference, or artifact rule.

The app already does the slow part. `prepare_recording` clips the fit window, drops samples outside the amplitude limit, and applies the reference. `welch_spectra` estimates power on accepted runs only. `fit_spectra` turns that into a `ChannelFit`. `fit_cycles` turns the same prepared signal into cycles and `BurstEpisode`s. New code calls those functions and keeps the per-window results they currently throw away.

Language on every new panel matches the Findings page. The sentences describe the measurement. They do not assign a diagnosis, an arousal label, or a treatment effect.

## What to leave in place

| Piece | Role |
| --- | --- |
| `Recording` | µV data, `sfreq`, channel names, `valid`, `rejected_spans`, `marks` |
| `FitSettings` and `FitSettings.key()` | Window, Welch segment, specparam knobs, burst thresholds, reference, amplitude limit |
| `prepare_recording` | The only preprocessing path |
| `welch_spectra` | Whole-window PSD. Linear µV²/Hz. Linear detrend inside each accepted run |
| `fit_spectra` / `ChannelFit` / `SpectrumFit` | One fit for the fit-bar window. Peaks are `(center Hz, height in log10 power, bandwidth Hz)` |
| `fit_cycles` / `CycleFit` / `BurstEpisode` | One channel, one band. Episode times are seconds from the start of the open recording |
| `compare_spectra`, `alpha_peak`, `band_power`, `band_reactivity_percent` | EC versus EO. A missing peak stays missing. A zero eyes-open denominator returns no percent |
| `accepted_runs`, `covered_length` | Contiguous accepted samples. Burst occupancy already merges overlaps |
| `draw_head` | Scalp maps. A `None` value is an uncolored site |
| `CallWorker` / `launch` | Every new fit runs off the UI thread, with a generation counter, the same way `SpectrumPanel` does |
| Tabs | Traces, Spectrum, Cycles, Compare, Findings |

`FitSettings.epoch_s` and `_fit_epochs` stay as they are. They cut the window into equal epochs and **collapse** them to median peaks and a mean spectrum. The sliding-window plot needs the list of fits before that collapse. Call `fit_spectra` once per window and store the list. Do not route the time series through `_combine_epochs`.

The trace scrollbar and the cycle scrollbar only move the picture. Refit is the fit bar. A click in a new graph moves the picture. It changes the fit window only when the user presses the existing Refit control, or a button that fills the fit bar and asks for Refit.

## Time base

Every new time axis is seconds from the start of the recording currently on screen, the same clock as `BurstEpisode.start_s` and `CycleFit.t_offset`.

`prepare_recording` crops to the fit window, so sample 0 inside a fit is not sample 0 of the file when the fit bar starts later than 0. Convert back with the window start:

```text
t_file = t_start + sample_in_prepared / sfreq
sample_in_file = round(t_file * sfreq)
```

`t_start` is the clipped fit-bar start (`window_bounds` in `analysis/settings.py`). A spectrogram of a fit window from 120 s to 180 s is labeled 120–180, and a click at 150 s seeks the trace to 150 s.

Store sample indices on results that get exported. Display seconds. Keep `settings.key()` on the result so a saved table can be matched to the fit that produced it.

Rejected samples stay rejected. A time-frequency or coherence estimate is computed on each `accepted_runs` stretch, then placed onto the file clock. Bins that sit on a rejected stretch are NaN. They are not zeros, and a taper is not fit across the gap.

## Shared inspection

Add this on `MainWindow` while building the spectrogram, not after the other graphs.

```python
@dataclass(frozen=True)
class Inspection:
    condition: str          # "ec" or "eo", the recording on screen
    channel: str
    t_center_s: float       # file clock
    span_s: float           # width of the spectrogram window, or the clicked burst
```

`MainWindow` emits one signal when it changes. Listeners:

- `TracePanel` sets its scroll so `t_center_s` is inside the visible page. It does not call `prepare_recording`.
- `CyclePanel` does the same.
- The spectrogram redraws a vertical line at `t_center_s`.
- A side readout lists what that instant already contains: the spectrogram column, any `BurstEpisode` on that channel that overlaps the span, and the cycle spans from the current `CycleFit` that overlap it.

If a sliding-window `ChannelFit` covers `t_center_s`, the readout draws that fit with the existing spectrum renderer. If none does, the readout says the Spectrum tab is still the fit-bar window and offers **Set fit window**, which writes start and length into the fit bar and leaves Refit to the user.

Channel selection stays the existing path: `_select` already updates the trace, spectrum, and cycles. The new tab uses that instead of a private channel index.

Cache key for any heavy result: recording identity (path, sample count, sampling rate), `FitSettings.key()`, channel name, and the control that is unique to that graph (spectrogram window and overlap, sliding-window length, coherence pair). Drop the cache when the recording is replaced or Refit runs.

## Where the first view lives

Add one tab, **Time**, rather than a tab per graph. `icons.TAB_ICONS` needs an entry; reuse an existing glyph until a new one is worth drawing. The tab holds a mode combo:

1. Spectrogram
2. Spectral trajectory (sliding specparam)
3. Burst raster
4. Reactivity

Later modes (coherence, morphology, duration, amplitude–frequency, quality) are further entries in that combo, or a second tab only when the mode needs its own pair of channels. Pre/post stays on Compare. Residuals stay on Spectrum. The exponent variability map stays on the Spectrum scalp-map control. PAC is its own panel and is absent from the default Refit.

Wire the tab in `MainWindow` the same way as the others: `set_recording` on open, `apply_settings` on Refit, `set_selected` on channel change, and a branch in `save_image` / `save_data`.

Export goes through `export.py`: one CSV writer per result, and the same columns included in Save session. `fit_metadata` gains the extra control values (spectrogram window, overlap, sliding length) when those results are saved.

Tests go in `tests/test_timefreq.py`, unittest style, synthetic NumPy, no window unless the test is an offscreen smoke test like `test_window_shows_fit`.

---

## 1. Spectrogram and burst overlay

**Purpose.** Show power across time and frequency for the fit window. The Spectrum tab remains the average of that window.

**Module.** `bifooof/analysis/timefreq.py`, function `spectrogram(recording, settings, channel_index, window_s=2.0, overlap=0.75)`.

**Steps.**

1. `prepared = prepare_recording(recording, settings)`.
2. Read one channel and its mask (`mask` and finite samples).
3. `nperseg = round(sfreq * window_s)`, at least 16. `noverlap = round(nperseg * overlap)`. Reject overlap outside `[0, 0.95]`.
4. For each accepted run long enough for `nperseg`:
   - linear `detrend`, matching `welch_spectra`
   - `scipy.signal.spectrogram` with `window="hann"`, `scaling="density"`, `mode="psd"`, `detrend=False`
   - place bin centers on the file clock
5. Color values are `10 * log10(power)` with power in µV²/Hz. Non-finite and empty bins are NaN.
6. Return times, freqs, power in dB, `nperseg`, `noverlap`, `df_hz = sfreq / nperseg`, and `dt_s = (nperseg - noverlap) / sfreq`.

Frequency limits default to `FitSettings.fmin` and `fmax`. The spectrogram window is **not** `FitSettings.segment_s`. The spectrum fit stays on its own Welch segment (default 4 s, and SciPy’s Welch overlap stays at half the segment). Changing the spectrogram control must not refetch the Spectrum tab.

**Panel.** Heatmap, time on x, Hz on y, dB on the color bar, labeled `dB re 1 µV²/Hz`. Caption states `df` and `dt`, and that a burst shorter than the window is spread across that window. Controls: channel (follows the shared selection), window length, overlap, frequency limits, and a color lock (auto, or fixed min/max). Overlay toggle draws the current channel’s `BurstEpisode`s as segments at `1000 / period_ms` when the period exists, spanning `start_s` to `end_s`. Episodes outside the fit window are already absent because `fit_cycles` uses the same settings.

Clicking the heatmap sets `Inspection` with `span_s = window_s`. The lower strip of this mode shows the raw samples of the selected channel for that span, taken from `Recording.data` (the on-screen recording, before the fit’s crop is confused with file time). Mark rejected samples on that strip. Beside it, list overlapping burst episodes: start, duration, cycle count, median amplitude, median frequency.

Run the spectrogram in a `CallWorker`. One channel per job. Switching channel starts a new generation and ignores the old result.

**Acceptance.** A 10 Hz sinusoid for the first half of a synthetic recording and a 20 Hz sinusoid for the second half. The early time bins peak near 10 Hz and the later bins near 20 Hz, within one frequency bin and one time step. A gap of invalid samples in the middle produces NaN columns over that gap.

## 2. Sliding-window specparam

**Purpose.** Show whether a change in the spectrum is the aperiodic background, a peak, or both.

**Module.** `sliding_spectra(recording, settings, window_s=30.0, overlap=0.5) -> list[SpectrumFit]` in `timefreq.py`.

**Steps.**

1. Prepare once.
2. Step windows of `window_s` by `window_s * (1 - overlap)` across the prepared recording. Drop a trailing stub shorter than the window.
3. For each window, crop with `Recording.cropped` and call `fit_spectra` with a child `FitSettings` whose `t_start` and `t_length` cover that crop only, and whose `epoch_s` is `None`. Copy the parent’s frequency range, segment length, aperiodic mode, peak limits, reference `"stored"`, and amplitude limit `0`, because prepare already applied those. This is the same idea as the child settings inside `_fit_epochs`.
4. Write file-clock `t_start` and `t_end` onto each returned `SpectrumFit`.
5. On `ValueError` (no accepted stretch, flat channel), keep a placeholder `SpectrumFit` with that channel `usable=False` and the reason. The time point still exists.
6. Attach `artifact_fraction`: invalid samples in the window divided by window length, from the prepared mask.

Default windows are 30 s with 50% overlap. Offer 10, 30, and 60 s. A 10 s window with a 4 s Welch segment has few averages; the caption says so. The existing rule that an epoch is at least 8 s still applies to `FitSettings.epoch_s` on the fit bar. The sliding control is separate and may be 10 s.

**Per window, per channel, read from `ChannelFit`:**

| Field | Source |
| --- | --- |
| Exponent, offset, knee | `exponent`, `offset`, `knee` |
| Peak center, height, bandwidth | peaks whose center lies in `settings.alpha_band()`. Tallest one. Empty stays empty |
| R², error | `r_squared`, `error` |
| Width-floor flag | `peak_at_width_floor` |
| Weak fit | `r_squared` below `R2_WEAK` (0.6) |

**Panel.** Three aligned time series for the selected channel: exponent, in-band peak height, in-band peak center. Offset can replace exponent from a control. A missing peak is a gap. A weak fit or a width-floor peak uses a different marker. Lines connect only consecutive windows that both have a value. The caption names the window length, the overlap, the Welch segment, and `aperiodic_mode`.

**Acceptance.** Synthetic data with a fixed aperiodic-like background and an alpha-band oscillation that starts halfway. The peak-height series is absent in the first half and present in the second. The exponent series does not jump by a large step at that boundary. (Build the carrier with `specparam.sim` or a 1/f-filtered noise plus a gated sinusoid. Allow the tolerance the estimator actually has; the test checks the step in exponent against the step in peak height.)

## 3. Burst raster

**Purpose.** When bursts happen, how long they last, and whether channels do it together.

**Data.** Call existing `fit_cycles` per selected channel. Do not write a second burst detector.

`BurstEpisode` already has `start_s`, `end_s`, `n_cycles`, `period_ms`, `volt_amp` on the file clock. The channel and band live on the parent `CycleFit`. Mean frequency is `1000 / period_ms` when `period_ms` is positive.

Extend episodes only if a field is missing at render time. Keep `burst_episode_rows` working. Adding a frequency column there is fine.

**Panel.** One horizontal row per selected channel, one segment per episode from `start_s` to `end_s`. Hover or a table shows cycle count, duration, median amplitude, median frequency. Click sets `Inspection` to the episode’s midpoint and `span_s` to its duration, and seeks the trace.

Default channel set is the current channel. A list lets the user add others. Each added channel is a worker job. Coincidence is drawn by shared time only. The caption states that overlapping times are not connectivity.

**Occupancy.** Use `CycleFit.burst_fraction`, which is `covered_length` of burst spans divided by accepted samples. `covered_length` already merges overlaps. Show it as a percent. Also show episode count, median duration, median amplitude, and the median gap between one episode’s `end_s` and the next episode’s `start_s` on that same channel.

**Acceptance.** Three gated bursts of known start, duration, and cycle count in one channel of noise-free or lightly noised data, band and thresholds loose enough that bycycle keeps them. Starts within one cycle period, durations within one cycle, cycle counts within 1 of the constructed count.

## 4. Cycle morphology

**Purpose.** See the shape of individual cycles, not only their median symmetry.

**Data.** `CycleFit.filtered`, `spans` `(last_trough, next_trough, is_burst)`, and `peaks`. `_cycle_rows` currently drops sample landmarks. Add `sample_last_trough`, `sample_next_trough`, and `sample_peak` to each cycle dict when those columns exist, so the CSV and the overlay share one table. Amplitude, period, `rise_decay` (`time_rdsym`), and `peak_trough` (`time_ptsym`) are already on the row.

**Steps for the selected cycles (all, or bursts only).**

1. Extract `filtered[start:stop]` for each span. Skip non-finite stretches.
2. Peak alignment shifts the snippet so `sample_peak` is at the same index. Trough alignment uses the trough-to-trough span as extracted.
3. Native-time view plots each snippet in seconds with its own length. The x-axis is time from the alignment point.
4. Normalized view resamples each snippet to a fixed sample count and labels the axis “normalized cycle”. The native view is the default.
5. Draw cycles in a thin low-opacity line. Overlay the median and the 10th–90th percentile at each aligned index. Snippets shorter or longer than the median keep their native length in the native view; the percentile envelope uses the indexes all snippets still cover.
6. A small set of histograms or a table: amplitude, period, rise/decay, peak/trough. Values are the ones bycycle stored.

**Acceptance.** A symmetric sine has median `time_rdsym` and `time_ptsym` near 0.5. A waveform with a fast rise and a slow decay sits away from 0.5 on rise/decay, on the side bycycle’s definition produces. Assert the direction with a comment that quotes the bycycle column, and check the overlay median has the same asymmetry.

## 5. Reactivity around eye events

**Purpose.** Show the time course of the selected band. Two open files stay a comparison of two fits. A transition curve exists only when one recording still contains the event times.

**Important current behavior.** `split_marked_conditions` cuts a marked file into separate EC and EO recordings and each side loses the other side. The continuous signal across the mark is the recording **before** that split. Keep that full recording available for this mode (the loader already has it in hand in `_queue_load` / the load success path before `_apply` splits it). Store it as an optional source on the session when both marks exist. The split copies remain what Traces, Spectrum, and Compare show.

**Two modes.**

- **Separate recordings.** EC and EO are two files, or a file that was split. Plot the existing comparison numbers (peak height difference, band power, percent when `band_reactivity_percent` returns a value, exponent difference). There is no time axis between the files and no invented transition.
- **Marked recording.** Marks whose labels `condition_in_text` recognizes are events. For each event, take a window around the onset on the unsplit recording (default 10 s before and 20 s after, configurable). Compute band power, in-band peak height, and exponent in shorter blocks inside that window (the sliding-window function, or Welch band power alone when the block is too short for specparam). Align the blocks to the onset and draw one line per event plus the median across events.

Posterior names (`O1`, `O2`, `Oz`, `Pz`) are the default selection when they exist. The channel control can select anything else.

Percent change uses a stated baseline: the median of the pre-event block. If that baseline is near zero, show the absolute difference and leave the percent blank, the same policy as `band_reactivity_percent`.

**Acceptance.** A sinusoid in the selected band that steps down at a known mark. The aligned band-power curve is high before the mark and lower after it. A session made of two already-split files does not draw that curve.

## 6. Coherence and phase

**Purpose.** Frequency-resolved relationship between two channels that share a recording.

**Module.** `bifooof/analysis/coherence.py`.

**Steps.**

1. Prepare once, so both channels have the same reference, crop, and amplitude mask.
2. Keep samples where **both** masks are true. Split with `accepted_runs`.
3. On each run, `scipy.signal.csd` and `welch` (or `scipy.signal.coherence`) with `nperseg` from `FitSettings.segment_s`, Hann window, and the same detrend as `welch_spectra`.
4. Average complex CSD and the two auto-spectra across runs, weighted by run length, matching `welch_spectra`.
5. Magnitude-squared coherence is `|Sxy|² / (Sxx Syy)`. Phase is the angle of `Sxy`. Imaginary coherence is `Im(Sxy) / sqrt(Sxx Syy)`.
6. If the accepted duration is shorter than a few Welch segments, return the curves and a note that the estimate is thin. The panel shows the note.

Weighted phase-lag index needs the CSD of each segment before the average. Add it as a second function in the same module once the segment CSD list exists. The first panel ships magnitude-squared coherence, phase, and imaginary coherence.

**Panel.** Three rows: coherence from 0 to 1, phase in degrees, and the two raw signals for the inspection span. Channel pickers default to the selected channel and the next posterior or the next selected row. Caption names the reference from the fit settings and says a high value is not anatomical connectivity. When magnitude-squared coherence is high and imaginary coherence is near zero at the same frequency, mark that frequency as zero-lag. The wording is “zero-lag coupling, which volume conduction can produce.”

Compare EC against EO only by computing the measure on each prepared recording with the same settings and overlaying the curves. Two files are never concatenated.

**Acceptance.** Two channels of a shared sinusoid plus a fixed phase shift show a coherence peak at that frequency. Two independent noise channels do not show a peak that survives a simple threshold across repeated seeds (assert the shared-tone peak is well above the noise-pair maximum in the same band).

## 7. Peak-frequency tracking

**Purpose.** Follow one oscillatory peak across the sliding windows.

**Data.** The in-band peaks from item 2. This is a renderer plus a small matcher, not a new fit.

**Matcher.** Walk windows in time. A peak continues a track when its center is within `max(1.5 Hz, half of the previous bandwidth)` of the track’s last center. If several peaks in the next window qualify, take the closest. If none qualify, the track ends. A later peak starts a new track. A window with no in-band peak ends the track. Draw each track as its own line. Marker area or color is peak height. Gaps are breaks.

The summary lists, for the selected track or the tallest track: center range, median height, median bandwidth, and the percent of windows in which an in-band peak was present.

**Acceptance.** A tone that moves from 9 Hz to 11 Hz across the recording stays one track inside the alpha band. A simultaneous stronger tone at 20 Hz does not capture the track. The test band is 8–12 Hz, so the beta tone is ineligible.

## 8. Burst duration distribution

**Purpose.** Short bursts versus long episodes.

**Data.** `CycleFit.episodes` for the selected channel. Duration is `end_s - start_s`.

**Panel.** Histogram, empirical cumulative curve, and a table: count, median, 25th and 75th percentile, episodes per minute of accepted time, occupancy from `burst_fraction`, median cycles per episode, and the inter-episode gaps from item 3. When both EC and EO cycle fits exist, overlay the two histograms with the same bin edges and the same burst settings. The fit bar already forces those settings to match.

**Acceptance.** Episodes built at known durations land in the expected bins, and occupancy matches `burst_fraction` within a small numerical tolerance.

## 9. Amplitude–frequency scatter

**Purpose.** One point per bycycle cycle.

**Data.** Each cycle row. X is `1000 / period_ms` (Hz). Y is `volt_amp` (µV). Color is burst versus not, or EC versus EO when both fits are loaded. A toggle hides non-burst cycles.

Optional ordinary-least-squares line and a percentile band are display only. The caption says the line is a description of the points.

**Acceptance.** Cycles of one frequency and several amplitudes form a vertical cloud: the frequency interquartile range is a small fraction of the amplitude range’s relative spread. A frequency sweep spreads along x.

## 10. Phase–amplitude coupling

**Purpose.** A research view of whether the phase of a slow band lines up with the amplitude of a faster band.

**Module.** `bifooof/analysis/pac.py`. Nothing in `fit_spectra`, `fit_cycles`, Findings, or the default Refit imports it. The Time tab grows a PAC mode that computes only when opened.

**Steps, per channel (or a stated pair, phase from one channel and amplitude from another).**

1. Prepare, then take the longest accepted run, or concatenate nothing: filter each run separately and skip runs that are too short. Record how much time was used.
2. Bandpass the phase range and the amplitude range with a filter that is stable at `sfreq` (the cycle code uses a 3rd-order Butterworth SOS and `sosfiltfilt`; use at least that care, and reject a band that reaches Nyquist).
3. Hilbert transform: phase from the slow signal, amplitude from the envelope of the fast signal.
4. Tort’s modulation index on the phase-amplitude histogram.
5. Surrogates: circularly shift the amplitude envelope by at least one slow-cycle length, repeated a fixed count (start at 200). Report the MI, the surrogate mean and spread, and the rank of the real MI.
6. A comodulogram is a grid of those MI values. The panel states how many cells were tested. A single starred cell is not a corrected result; if a threshold is drawn, say whether it is per-cell or corrected across the grid.

Skip the fit and say why when accepted time is under 60 s, when either band is empty, or when the faster band is a low harmonic of the slower band and the waveform is strongly nonsinusoidal (`time_ptsym` or `time_rdsym` far from 0.5 on the current cycle fit). That skip is a displayed limitation, not a silent zero.

**Acceptance.** An amplitude-modulated synthetic (slow phase, fast amplitude locked to one phase) yields an MI above its surrogate distribution. An uncoupled pair of rhythms does not, across a handful of seeds.

## 11. Exponent variability map

**Purpose.** The Spectrum scalp map shows one exponent. This shows the spread across sliding windows.

**Data.** Item 2, every channel that `fit_spectra` already returns.

**Panel.** Reuse `draw_head`. A combo switches between the median exponent and the median absolute deviation across windows. A site with fewer than 3 usable windows is `None` and draws as an empty site, with its usable count available in the table. The color scale is shared when EC and EO are both open. The table lists usable windows and `artifact_fraction` per channel.

**Acceptance.** Two synthetic channels with the same stationary spectrum produce exponents within a small tolerance of each other, and a variability well below the difference caused by actually changing the exponent on a third channel.

## 12. Pre/post comparison

**Purpose.** The same paired comparison as EC/EO, with labels that match the experiment the user is running.

**UI.** On Compare, a mode combo: Eyes closed / Eyes open, Pre / Post, Custom. Custom asks for two short names. The two open buttons and `recordings["ec"]` / `recordings["eo"]` stay the storage slots. The mode only changes words, which file is “left”, and which file is “right”. Default remains EC on the left and EO on the right. Pre/post puts the first-opened slot on the left and the second on the right, and the panel says the order is the order the files were opened.

**Calculation.** Add `compare_pair(left, right, band)` that returns the same numbers `compare_spectra` already computes, with neutral names. `compare_spectra` calls it and keeps the current EC/EO field names so export and the Findings report stay valid. Display code maps names through the mode. Percent change still goes through `band_reactivity_percent` (no percent when the right-hand denominator is missing or not positive) and the absolute difference is always shown beside it when both sides exist.

Paired dots or connected lines use one point per shared channel name. A difference scalp map uses `draw_head` on channels present and usable in both fits. A channel on only one side is absent from the map and listed in the table.

The caption says a difference between two sessions is not a treatment effect.

Burst occupancy difference is `burst_fraction` in percentage points when both cycle fits exist for that channel. Median burst duration comes from the episodes. R² and error stay the per-recording `ChannelFit` values.

**Acceptance.** `compare_pair` on two fits of the same recording yields differences at numerical noise. Peak-height difference is `None` when either side has no in-band peak.

## 13. Signal-quality timeline

**Purpose.** Show which stretches the fits are allowed to use, and let the user add or remove exclusions without rewriting the file.

**What already exists.** Non-finite samples, MNE `BAD` annotations, bad channels, and `reject_large_amplitudes` (the fit-bar amplitude limit). Those already flow through `Recording.valid` and `rejected_spans`.

**Module.** `bifooof/analysis/quality.py` proposes extra spans. Each span has start, stop, channel or “all”, a kind, and a level.

| Kind | Level | Rule of thumb |
| --- | --- | --- |
| Amplitude | detected | Already the amplitude limit. Display those spans; do not detect them twice |
| Flat | detected | Window of at least 0.5 s whose standard deviation is near zero, or a long run of identical values |
| Clip | detected | A run stuck at a repeated extreme value |
| Step | detected | A first difference far above that channel’s own typical difference |
| Muscle | suspected | A short window whose power above 30 Hz is several times that channel’s own median |
| Drift | suspected | A window whose linear slope or power below 1 Hz is extreme for that channel |
| Mains | suspected | A narrow Welch peak at 50 Hz or at 60 Hz, clearly above the neighboring bins |
| File mark | confirmed | Existing `rejected_spans` and bad-sample masks |
| User | confirmed | Spans the user draws |

Suspected spans are drawn and listed. They change `valid` only after the user confirms them. Confirm and restore edit a session mask held on `MainWindow`, then build a copied `Recording` via the same pattern as `_copied` in `prepare.py`: new `valid`, extra `rejected_spans`, a note. The file on disk is untouched. Refit runs on the copy. Restore removes the session span and refits.

**Acceptance.** Synthetic clip, flatline, and a large transient each produce a detected span covering the event. A clean sinusoid produces no detected span.

## 14. Spectral residuals

**Purpose.** Show where the specparam model leaves the Welch spectrum.

**Data.** Already on `ChannelFit`: `spectrum` and `model` are log10 power. Residual is `spectrum - model`. R², error, `peak_at_width_floor`, `fmin`, and `fmax` are already there.

**UI.** A residual axes under the current spectrum plot in `render_spectrum` / `SpectrumPanel`, with a zero line. Clicking a residual frequency selects that frequency in the readout (the nearest bin). No new fit.

**Acceptance.** A spectrum synthesized from an aperiodic component plus one Gaussian peak, then fit with a compatible model, has a small residual RMS. The same spectrum fit with the peak disallowed (`max_n_peaks` enough but the peak outside the fit range, or a model limited so the peak cannot be represented) leaves a residual bump at the peak frequency. Use `specparam`’s own simulation helpers if they are already a dependency, so the residual test matches the model class the app fits.

## 15. Linked cursor, finished

Item 1 puts the cursor on the spectrogram, the trace, the cycles, and the local readout. Finish it as the other modes appear:

- Spectral trajectory, raster, reactivity, coherence, and quality all draw the same `t_center_s` and set `Inspection` when clicked.
- The readout’s spectrum block prefers the sliding-window `ChannelFit` that contains `t_center_s`.
- The readout’s cycle block filters `CycleFit.spans` and `episodes` to the span.
- **Set fit window** writes the span into the fit bar. The user still presses Refit. That keeps the scrollbar rule the app already has.

A spectrogram click at 120 s therefore shows the raw samples around 120 s, the bursts that overlap that window, and the specparam fit of the sliding window that contains 120 s, when that fit has been computed.

---

## Build order

Each step is independently usable and keeps the previous tests passing.

1. **Time tab, spectrogram, burst overlay, inspection cursor, raw strip.** Seeks Traces and Cycles. Exports the spectrogram grid and the episodes it drew.
2. **Sliding specparam** on that tab, and the readout starts showing the window fit. Peak-center gaps included.
3. **Reactivity mode**, including retention of the unsplit marked recording.
4. **Compare mode** for Pre/Post and Custom, via `compare_pair`.
5. **Coherence** mode (magnitude-squared, phase, imaginary). Segment CSD retained so wPLI can follow.
6. **PAC** module and mode, surrogates included, off the default Refit path.

After those, in any order, because each is a view of data the earlier steps already store:

- Burst raster and duration histogram (episodes already exist; raster can also ship inside step 1 if the overlay is not enough).
- Cycle morphology, after the cycle rows keep sample landmarks.
- Peak tracking on the sliding-window series.
- Amplitude–frequency scatter from cycle rows.
- Exponent variability on `draw_head`.
- Residuals under the current spectrum plot. This one does not depend on the Time tab and can land at any time.
- Quality timeline and the session mask. Do this before trusting long unattended batch fits on new files; it is not required for the spectrogram.

## Files a change is allowed to touch

- `bifooof/analysis/timefreq.py`, `coherence.py`, `pac.py`, `quality.py` — new.
- `bifooof/analysis/cycles.py` — add sample columns to cycle rows. Leave burst detection alone.
- `bifooof/analysis/compare.py` — add `compare_pair`. Keep `compare_spectra` fields.
- `bifooof/analysis/specparam_fit.py` — call it. Avoid changing the collapsed epoch result.
- `bifooof/plots.py` and `bifooof/ui/` — rendering and the Time tab. Residuals extend the spectrum figure.
- `bifooof/ui/main_window.py` — tab, inspection signal, save branches, optional unsplit recording.
- `bifooof/export.py` — CSV rows and metadata keys.
- `tests/test_timefreq.py` — the acceptance tests named above.

## Out of scope

Clinical classification, automatic EO/EC labeling of an unmarked file, connectivity claims, interpolation across missing electrodes, and a second preprocessing path.
