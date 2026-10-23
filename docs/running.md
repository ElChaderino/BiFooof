# The window

This file is what BiFooof does while it is open. When a control, a caption, or a saved column changes, change this file in the same edit. [additional-graphs.md](additional-graphs.md) is the plan the Time views were built from. Where the two disagree, this file matches the program.

Times on every page are seconds from the start of the recording on screen. A click moves the picture. **Refit** is the only control that refits. **Set fit window** writes a stretch into the fit bar and leaves Refit to you.

The scrollbars on Traces and Cycles only move the picture. Rejected samples stay out of the fits. A missing specparam peak stays missing. It is not entered as zero.

The command line fits spectra, cycles, and the eyes-closed / eyes-open comparison. It does not compute the Time views.

## Pages

Traces, Spectrum, Time, Cycles, Compare, Findings. The channel you select is the channel on all of them.

### Traces

The raw recording, in microvolts, for the page length and gain on that page. A click on Time scrolls this page so that time is in view. It does not change the fit window.

### Spectrum

specparam on the fit-bar window. The curve is the Welch spectrum, the model, and the aperiodic part, in log10 power. Under it is the residual, spectrum minus model, on the same frequency axis. Click either axes and the caption reports the residual at the nearest bin.

The head maps sit to the right of the spectrum, exponent above peak frequency, large enough to read every site. A weak fit, R² below 0.6, is ringed. A peak at the narrowest allowed width is marked. A site with no fit is empty.

Epochs, 30 s or 60 s, still collapse to one spectrum. Peaks within 1 Hz are grouped. Center, height, and width are medians of the epochs that found the peak. An epoch that missed it is left out of the median. The Time tab does not use that collapse. Each sliding window stays its own fit.

### Cycles

bycycle on the selected channel and the band on the fit bar. The page names the cutoff when cycles are found and none of them form a burst. Burst cycles are cycles that passed the burst rule. Burst episodes are the separate runs of those cycles. A cycle cannot cross a rejected gap.

A click on Time scrolls the filtered trace to that time.

### Compare

Eyes closed is the file opened with the eyes-closed button. Eyes open is the other. Both fits keep running.

The pair menu only changes words:

- **Eyes closed / Eyes open.** The usual captions, EC and EO.
- **Pre / Post.** Pre is the file opened as eyes closed. Post is the file opened as eyes open. The caption says a difference between sessions is not a treatment effect.
- **Custom.** Two names you type. The same sentence, for any two recordings.

Peak Δ is left-hand fitted peak height minus right-hand fitted peak height, in log10 power. Each side keeps its own tallest peak in the band. Δ band % is `(left − right) / right × 100` for mean Welch power in the band. It stays blank when the right-hand power is missing or not positive. The figure matches Spectrum: the two spectra are on the left, with one line per channel of in-band peak height underneath, and the head maps of the exponent difference and the peak-height difference are on the right. The cycle measures and the channel table sit beside the figure, not under it. A side with no peak is a dot by itself. A missing peak is not drawn as zero.

Findings uses those same names. The default is eyes closed and eyes open. Pre / Post and a custom pair are written with the names on Compare. It restates measurements. It does not assign a diagnosis.

### Time

One tab, one menu. The default is the spectrogram, so opening a file does not start a sliding specparam fit or a coupling estimate. Every view uses the fit bar: window, reference, amplitude limit, frequency range, and rejected samples. The spectrogram window is not the Welch segment. Welch stays at whatever the fit bar says, default 4 s.

**Spectrogram.** Power for the selected channel, Hann window, in dB relative to 1 µV²/Hz. The choices are 1, 2, or 4 s, and 50% or 75% overlap. The default is 2 s and 75%. The caption states the frequency bin and the time step, and that a burst shorter than the window is spread across that window. Frequency limits are the fit bar. **Bursts** draws the current channel’s burst episodes at `1000 / period` Hz when the cycle fit has a period. **Lock color** holds the 2nd-to-98th percentile range and shows two dB boxes. Typing in those boxes keeps that range. Unchecked, the range is chosen again for the next picture. A rejected stretch is a blank column, not a bridge. Click and the lower strip shows the raw samples around that time. Rejected samples on the strip are marked. The caption names the strongest bin in that column, burst episodes that overlap the click (start, duration, cycles, amplitude, and frequency), and how many cycles from the cycle fit fall in the stretch. If a sliding fit already covers that time, the same click draws that window’s spectrum under the strip. If none does, that row says the Spectrum tab is still the fit-bar window.

**Exponent and peaks.** specparam on successive windows. The choices are 10, 30, or 60 s, and no overlap or 50%. The default is 30 s and 50%. A 10 s window with a 4 s Welch segment has few averages, and the caption says so. The caption also names the Welch segment. The top trace is exponent, offset, or knee. A fixed slope has no knee, so that trace is a gap. Under it are in-band peak height and peak frequency. Marker size follows peak height. A missing peak is a gap. An open circle is a weak fit or a peak at the narrowest allowed width. Lines follow each peak only while its center stays within 1.5 Hz, or half its previous bandwidth, of the last point. A window with no in-band peak ends the track. The caption gives the tallest track’s frequency range, median height, median width, and the percent of windows with any in-band peak. Click a time and the lower axes show that window’s spectrum, model, and aperiodic fit. If the click falls outside every window, the caption says the Spectrum tab is still the fit-bar window. **Every channel** fits all channels and adds one head map, either the median exponent or its spread. A site with fewer than three usable windows is left empty. The table lists usable windows and the rejected fraction for each channel. When Every channel is on and another file is open, that file is fit with the same windows and the two maps share one color scale. This is not epoch mode.

**Bursts.** One row per channel, one bar per burst episode. The default is the selected channel. A checked list picks other channels. **Every channel** fits the rest and ignores the list. The table gives channel, start, duration, cycles, amplitude, and frequency. Bars that line up in time are not a connection between channels. A click on a bar uses that episode’s duration as the fit-window length it will offer.

**Burst lengths.** Histogram and cumulative curve of episode duration for the cycle fit already on the Cycles page. When both recordings have a cycle fit for this channel, both histograms use the same bins. The caption gives count, median, the 25th to 75th percentile, episodes per minute of accepted time, occupancy from the burst fraction, median cycles per episode, and the median gap.

**Cycle shape.** Each cycle from the current cycle fit, drawn faint, with the median and the 10th to 90th percentile. **Trough** or **Peak** sets the alignment. **Normalized** resamples trough to trough onto one axis from 0 to 1. **Bursts** hides cycles that are not in a burst. Under the overlay are histograms of amplitude, period, rise/decay, and peak/trough. The native view is the default.

**Amplitude and frequency.** One point per cycle. Frequency is `1000 / period` in Hz. Amplitude is the cycle’s voltage. With one recording, burst cycles are a separate color. With both recordings, color is the recording. **Bursts** hides cycles that are not in a burst. A line and a band describe the points on screen. The caption says the line is not a cause.

**Reactivity.** Mean Welch power in the fit-bar band, in 2 s blocks. The channel menu starts on O1, O2, Oz, or Pz when the file has one and the current channel is not already posterior. If the original file had eye marks, those onsets are the events. The default is 10 s before and 20 s after, and both can be changed. One line per mark and the median, for band power. When the window is at least 8 s, the same alignment is drawn for in-band peak height and exponent. The split eyes-closed and eyes-open sides do not contain the crossing, so this uses the recording from before the split. Two files opened separately are drawn side by side. The caption says there is no transition between them, and it adds the peak-height difference, the exponent difference, and the band-power percent when those fits exist. One file with no eye marks is a single trace and the cursor.

**Coherence.** The selected channel and the channel in the second menu. The second channel starts as the next posterior site, or the next row. Both go through the same reference and the same rejected samples. Three traces: magnitude-squared coherence, phase in degrees, and imaginary coherence. wPLI is drawn when there are at least five segments and only one file is drawn. When both recordings contain the pair, both coherences are overlaid and the files are not joined. A lower row shows the two raw signals for the clicked stretch. Click that row to choose a time. Orange marks a bin where coherence is at least 0.5 and the imaginary part is near zero. The caption calls that zero-lag coupling, which volume conduction can produce. A short accepted stretch says the estimate is thin.

**Coupling.** Phase–amplitude coupling. Phase stays on the selected channel. Amplitude can be that channel or another. It runs when you open this view, not when you press Refit. The default bands are 4–8 Hz for phase and 30–45 Hz for amplitude. Surrogates are 200 time shifts of the amplitude. A perfectly periodic modulation stays high after a time shift, because the shift only moves which phase bin is tall. The recording needs about 60 s of accepted signal; otherwise the page says so and does not invent an index. If the amplitude band contains a harmonic of the phase band, the index is skipped only when the current cycles are far from symmetric: rise/decay or peak/trough at least 0.15 from one half. Otherwise the index is still shown, and the caption says so. **Map** draws a coarse grid of the index and states how many cells were tested. Surrogates are for the selected bands only, and the map is not corrected across cells.

**Quality.** Stretches the rules flag, and a way to leave them out of later fits. The file on disk is not rewritten. Dragging across a channel on the timeline adds a confirmed span for that channel. **Put back** removes it. The file is not rewritten.

| Kind | What it means |
| --- | --- |
| Clip, flat, step, amplitude | Detected. The rule is sure. |
| Muscle, drift, mains | Suspected. Listed, and left in the fit until you choose otherwise. |
| File mark, missing | Already on the file, or non-finite samples. |
| Left out | A stretch you confirmed on this page. |
| Drawn | A span dragged on this page. It is already left out. |

**Leave out** copies the recording with that stretch removed and refits. The channel you were on stays selected. **Put back** restores a stretch that was left out here. Detected and suspected rows do nothing until Leave out.

## Saving

**Save image** writes the current page as a PNG. Findings is text; use Copy on that page.

**Save data** writes the current table.

- Spectrogram: one row per time-frequency bin that has a value, with the settings key, channel, time, sample index, frequency, and dB.
- Exponent and peaks: one row per window, with the settings key, sample index, exponent, offset, knee, peak, bandwidth, R², and the fraction of the window that was rejected.
- Bursts: one row per episode, with duration, cycles, amplitude, and frequency.
- Burst lengths: one summary row.
- Cycle shape and amplitude and frequency: one row per cycle, with the settings key and the sample indexes stored on the cycle.
- Quality: one row per span.
- Reactivity, when it is a single trace: time and band power.

**Save session** writes the traces, each condition’s spectrum, peaks, and metadata, the cycle tables, the comparison, findings, and `time.png` / `time.csv` when the Time page has rows. The metadata names the spectrogram window and overlap once that picture exists, and the sliding window once that fit has been run.
