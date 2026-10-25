# The window

This file is what BiFooof does while it is open. When a control, a caption, or a saved column changes, change this file in the same edit. [additional-graphs.md](additional-graphs.md) is the plan the Time views were built from. Where the two disagree, this file matches the program.

Times on every page are seconds from the start of the recording on screen. A click moves the picture. **Refit** applies staged fit settings. **Set fit window** writes a stretch into the fit bar and leaves Refit to you. The selected band updates its dependent views immediately.

MNE annotations use that recording-local clock, including cropped FIF recordings with a nonzero first sample. Ordinary event and eye-state annotations are retained. Duration-bearing WinEEG artifact marks are excluded from fits. Clock notes do not end an eye-state span. CSV time columns must be finite, increasing, and uniformly sampled; timestamp gaps or rate changes produce an error instead of being compressed. Timestamp rounding within 1% of the sample interval, or 1 microsecond, is accepted.

The scrollbars on Traces and Cycles only move the picture. Rejected samples stay out of the fits. A missing specparam peak stays missing. It is not entered as zero.

The command line fits spectra, cycles, and the eyes-closed / eyes-open comparison. It does not compute the Time views.

## Controls and feedback

The fit bar keeps window, spectrum preset, band, reference, epoch length, and amplitude limit visible. Its groups wrap at smaller window sizes. **More settings** opens the frequency limits, Welch segment, aperiodic model, peak limits, and burst rules; **Hide settings** folds them away.

Staged edits show **Settings changed** and **Refit\***. Reverting the edit clears that message. The band updates immediately. Invalid ranges show a specific message and disable Refit until corrected. Refit is also disabled while the spectral or cycle analyses are running. A small status-bar progress indicator shows background work.

Time keeps the view and channel on its first row. The controls for that view are labeled and wrap underneath. Long source names shorten in the header; the tooltip retains the complete description.

Keyboard shortcuts: **Ctrl+O** opens eyes closed, **Ctrl+Shift+O** opens eyes open, **Ctrl+R** refits, **Ctrl+S** saves a project, **Ctrl+P** opens a project, **Ctrl+Shift+E** exports results, **Ctrl+Shift+S** saves the current image, and **Ctrl+E** saves current data. **Ctrl+B** adds a bookmark; **Ctrl+[ / Ctrl+]** navigate bookmarks; **Ctrl+Alt+Z / Ctrl+Alt+Y** undo/redo review edits; **Esc** requests cancellation; **F1** opens the measurement guide.

File inspection and save errors show a short error with expandable technical details. Closing during analysis keeps the event loop responsive and closes automatically after running workers finish.

## Reference, montage, and amplitude

The fit bar and the trace page do different jobs. **Refit** rebuilds spectrum, cycles, and findings from the fit-bar reference. The trace montage changes only the picture on that page. The trace status line names both, and it says that the amplitude limit is applied on the fit and not on the trace.

The limit is in microvolts on the original samples, before the transform. The default is 500. A sample is dropped when it rises more than that far above the channel's straight trend. A constant offset is kept. `0` keeps every sample. There is no separate limit in µV/m². Rejected samples stay at their original times. Welch and cycle runs do not join the samples on either side of a gap. Runs of unequal length are weighted by the number of Welch windows each run contributes, at the usual 50% overlap, not by the number of samples.

`--reference` accepts `stored`, `average`, `linked`, `mastoid`, `left`, `right`, `vertex`, `ipsilateral`, `contralateral`, `rest`, `hjorth`, `csd`, and `global`.

**As stored** leaves the file reference in place.

**Average** uses one fixed set for the window: the analysis channels that have any accepted sample. The mean is subtracted only where every member of that set is finite. One missing contributor blanks that sample, including when a single channel is all that remains. A recording that never has two analysis channels stays stored.

**Linked ears** use A1 and A2. Those ears are also accepted under the names LE/EL and RE/ER. **Linked mastoids** use M1 and M2, also named T9 and T10. Ears and mastoids are not substitutes, and a channel label is not checked against the acquisition notes to decide whether A1 was an earlobe or a mastoid. A missing pair, or one blank side of a linked pair, keeps the stored reference and the note says so. One ear is not filled in from the other.

**Left** and **right** subtract that one ear. **Ipsilateral** and **contralateral** send a lateral channel to the ear its output depends on. Midline channels need both ears.

**Vertex** on a fit subtracts Cz and then leaves Cz blank, so Cz is not fitted as a channel. The trace vertex montage leaves Cz as the zero line.

**REST** is an infinity reference from a template spherical forward model. The note begins "REST — template geometry/spherical model." The values stay in microvolts. Fewer than four positioned 10-20 channels, a failed forward solution, or a channel set that never shares a finite sample keeps the stored signal and names the reason. The average reference is not put in its place. REST is not a trace montage. A second application is refused.

**Local Laplacian** subtracts the equal mean of a site's listed neighbors. The weights are equal, with no distance scaling, and the result stays in microvolts. A center is used only when every listed neighbor is in the file. A missing sample on a required neighbor blanks that instant. It is fit-only. A second application is refused.

**CSD** is the spherical-spline surface Laplacian, with MNE's defaults: stiffness 4, 50 Legendre terms, and lambda2 of 1e-5. Four analysis channels on the standard 10-20 layout is the minimum the spline will calculate. Fewer than eight adds the note that the surface is sparse. Units are µV/m², and the spectrum is µV²/(m⁴·Hz). A missing sample on any contributor blanks that time on every CSD channel. A second application is refused. The trace page can show the same montage, with its own µV/m² gain. That picture still waits for **Refit** before the fit changes.

**Global scale** is the average reference followed by one factor. The factor is the geometric mean of positive channel-by-frequency Welch power inside the fit band. Nonpositive bins, and bins at or below the larger of 1e-12 and 1e-8 times the peak bin, stay out of that mean. Samples are divided by the square root of the factor, so linear spectra divide by the factor. The aperiodic offset decreases by log10 of the factor. Exponent, peak center frequency, bandwidth, and log-power peak height above the aperiodic background are unchanged apart from ordinary numerical differences in the fit. Each recording is scaled on its own. The factor is stored with the fit. Labels are scaled µV and scaled µV²/Hz. If the factor cannot be estimated, the average reference is kept and the note says so. A second application is refused. This scale factor is not a regression and not a principal-component removal of a global signal.

Captions and metadata keep the reference requested, the reference applied, and the fallback note when those two differ. An eyes-closed / eyes-open pair names both applied transforms when they differ.

### Trace montages

As stored, average, vertex, ipsilateral, contralateral, left ear, right ear, longitudinal bipolar, transverse bipolar, circumferential bipolar, and current source density. Linked ears, linked mastoids, REST, the local Laplacian, and global scale are fit-only.

A bipolar pair is drawn only when both channels are checked. The label is the first site minus the second. Display band and notch run after the montage. The default band is Off. A stretch shorter than the filter can use is named, and the gap is not joined. Samples that would draw past the channel spacing are marked on the clipped line.

## Pages

Traces, Spectrum, Time, Cycles, Compare, Findings, Review, Study. The channel you select follows you between the analysis pages.

### Traces

The recording, in microvolts, for the page length and gain on that page. A click on Time scrolls this page so that time is in view. Left/Right and Page Up/Page Down page the trace; Jump moves to an explicit time. Drag to select an interval: the ruler reports time and the displayed-sample amplitude change on the channel where the drag began. Excluded endpoint samples have no amplitude measurement.

Display bandpass and notch, and the trace montages listed above, affect the picture. They do not change fit settings. The average display montage uses the same fixed contributor set as the fit. Supported bipolar pairs must both be visible. Brief extrema and missing-sample gap boundaries survive display reduction. The fit window, review bookmarks and exclusions can appear as overlays. The page states the montage, the display band, and the notch above the stack.

### Spectrum

specparam on the fit-bar window. The curve is the Welch spectrum, the model, and the aperiodic part, in log10 power. Under it is the residual, spectrum minus model, on the same frequency axis. Click either axes and the caption reports the residual at the nearest bin.

The head maps sit to the right of the spectrum, exponent above peak frequency, large enough to read every site. Names sit on a light halo. A weak fit, R² below 0.6, is a diamond. The selected channel is ringed. A peak at the narrowest allowed width is marked. A site with no fit is gray. Classic bands are tinted and named on the spectrum. The residual is drawn around a zero line. Comparison maps of exponent and peak height are centered on zero. The residual has its frequency label underneath; comparison titles use separate lines to stay readable in smaller windows.

Epochs, 30 s or 60 s, still collapse to one spectrum. Peaks within 1 Hz are grouped. Center, height, and width are medians of the epochs that found the peak. An epoch that missed it is left out of the median. The Time tab does not use that collapse. Each sliding window stays its own fit.

The epoch spectrum and model curves use the arithmetic mean of linear power, displayed in log10 units. Band-power comparisons therefore use mean linear Welch power. Parameter medians and peak counts remain separate from this curve average.

### Cycles

bycycle on the selected channel and the band on the fit bar. The page names the cutoff when cycles are found and none of them form a burst. Burst cycles are cycles that passed the burst rule. Burst episodes are the separate runs of those cycles. A cycle cannot cross a rejected gap.

A click on Time scrolls the filtered trace to that time.

### Compare

Eyes closed is the file opened with the eyes-closed button. Eyes open is the other. Both fits keep running.

Whole-recording comparisons use the shorter duration on both sides in the window and CLI, including the cycle fits. Explicit start/length choices are preserved. Replacement fits invalidate the previous comparison immediately; failed or pending results are not available to export.

The pair menu only changes words:

- **Eyes closed / Eyes open.** The usual captions, EC and EO.
- **Pre / Post.** Pre is the file opened as eyes closed. Post is the file opened as eyes open. The caption says a difference between sessions is not a treatment effect.
- **Custom.** Two names you type. The same sentence, for any two recordings.

Peak Δ is left-hand fitted peak height minus right-hand fitted peak height, in log10 power. Each side keeps its own tallest peak in the band. Δ band % is `(left − right) / right × 100` for mean Welch power in the band. It stays blank when the right-hand power is missing or not positive. The figure matches Spectrum: the two spectra are on the left, with one line per channel of in-band peak height underneath, and the head maps of the exponent difference and the peak-height difference are on the right. The cycle measures and the channel table sit beside the figure, not under it. A side with no peak is a dot by itself. A missing peak is not drawn as zero.

Findings uses those same names. The default is eyes closed and eyes open. Pre / Post and a custom pair are written with the names on Compare. It restates measurements. It does not assign a diagnosis.

### Time

One tab, one menu, and a channel menu. The channel menu is the same channel as Traces, Spectrum, Cycles, and Compare. The default view is the spectrogram, so opening a file does not start a sliding specparam fit or a coupling estimate. Every view uses the fit bar: window, reference, amplitude limit, frequency range, and rejected samples. The spectrogram window is not the Welch segment. Welch stays at whatever the fit bar says, default 4 s.

**Spectrogram.** Power for the selected channel, Hann window, in dB relative to 1 of the applied power unit. That unit is µV²/Hz, µV²/(m⁴·Hz) after CSD, or scaled µV²/Hz after global scale. The choices are 1, 2, or 4 s, and 50% or 75% overlap. The default is 2 s and 75%. The caption states the frequency bin and the time step, and that a burst shorter than the window is spread across that window. Frequency limits are the fit bar. **Bursts** draws the current channel’s burst episodes at `1000 / period` Hz when the cycle fit has a period. **Lock color** holds the 2nd-to-98th percentile range and shows two dB boxes. Typing in those boxes keeps that range. Unchecked, the range is chosen again for the next picture. A rejected stretch is a blank column, not a bridge. Click and the lower strip shows the raw samples around that time. Rejected samples on the strip are marked. The caption names the strongest bin in that column, burst episodes that overlap the click (start, duration, cycles, amplitude, and frequency), and how many cycles from the cycle fit fall in the stretch. If a sliding fit already covers that time, the same click draws that window’s spectrum under the strip. If none does, that row says the Spectrum tab is still the fit-bar window.

**Exponent and peaks.** specparam on successive windows. The choices are 10, 30, or 60 s, and no overlap or 50%. The default is 30 s and 50%. A 10 s window with a 4 s Welch segment has few averages, and the caption says so. The caption also names the Welch segment. The top trace is exponent, offset, or knee. A fixed slope has no knee, so that trace is a gap. Under it are in-band peak height and peak frequency. Marker size follows peak height. A missing peak is a gap. An open circle is a weak fit or a peak at the narrowest allowed width. Lines follow each peak only while its center stays within 1.5 Hz, or half its previous bandwidth, of the last point. A window with no in-band peak ends the track. The caption gives the tallest track’s frequency range, median height, median width, and the percent of windows with any in-band peak. Click a time and the lower axes show that window’s spectrum, model, and aperiodic fit. If the click falls outside every window, the caption says the Spectrum tab is still the fit-bar window. **Every channel** fits all channels and adds one head map, either the median exponent or its spread. A site with fewer than three usable windows is left empty. The table lists usable windows and the rejected fraction for each channel. When Every channel is on and another file is open, that file is fit with the same windows and the two maps share one color scale. This is not epoch mode.

**Bursts.** One row per channel, one bar per burst episode. Bar color follows that episode’s amplitude when the fit has one. The default is the selected channel. A checked list picks other channels. **Every channel** fits the rest and ignores the list. The table gives channel, start, duration, cycles, amplitude, and frequency. Bars that line up in time are not a connection between channels. A click on a bar uses that episode’s duration as the fit-window length it will offer.

**Burst lengths.** Filled histogram and cumulative curve of episode duration for the cycle fit already on the Cycles page. A dotted line marks each recording’s median duration. When both recordings have a cycle fit for this channel, both histograms use the same bins. The caption gives count, median, the 25th to 75th percentile, episodes per minute of accepted time, occupancy from the burst fraction, median cycles per episode, and the median gap.

**Cycle shape.** Each cycle from the current cycle fit, drawn faint, with the median and the 10th to 90th percentile. **Trough** or **Peak** sets the alignment. **Normalized** resamples trough to trough onto one axis from 0 to 1. **Bursts** hides cycles that are not in a burst. Under the overlay are histograms of amplitude, period, rise/decay, and peak/trough. The native view is the default.

**Amplitude and frequency.** One point per cycle. Frequency is `1000 / period` in Hz. Amplitude is the cycle’s voltage. With one recording, burst cycles are a separate color. With both recordings, color is the recording. **Bursts** hides cycles that are not in a burst. A line and a band describe the points on screen. The frame is the 2nd to 98th percentile, so one extreme cycle does not flatten the cloud. The line still uses every finite point. The caption says the line is not a cause.

**Reactivity.** Mean Welch power in the fit-bar band, in 2 s blocks. The channel menu starts on O1, O2, Oz, or Pz when the file has one and the current channel is not already posterior. If the original file had eye marks, those onsets are the events. The default is 10 s before and 20 s after, and both can be changed. One line per mark and the median, for band power. When the window is at least 8 s, the same alignment is drawn for in-band peak height and exponent. The split eyes-closed and eyes-open sides do not contain the crossing, so this uses the recording from before the split. Two files opened separately are drawn side by side. The caption says there is no transition between them, and it adds the peak-height difference, the exponent difference, and the band-power percent when those fits exist. One file with no eye marks is a single trace and the cursor.

**Coherence.** The selected channel and the channel in the second menu. The second channel starts as the next posterior site, or the next row. Both go through the same reference and the same rejected samples. Three traces: magnitude-squared coherence, phase in degrees, and imaginary coherence. wPLI is drawn when there are at least five segments and only one file is drawn. When both recordings contain the pair, both coherences are overlaid and the files are not joined. A lower row shows the two raw signals for the clicked stretch. Click that row to choose a time. Orange marks a bin where coherence is at least 0.5 and the imaginary part is near zero. The caption calls that zero-lag coupling, which volume conduction can produce. A short accepted stretch says the estimate is thin.

**Coupling.** Phase–amplitude coupling. Phase stays on the selected channel. Amplitude can be that channel or another. It runs when you open this view, not when you press Refit. The default bands are 4–8 Hz for phase and 30–45 Hz for amplitude. Surrogates are 200 time shifts of the amplitude. A perfectly periodic modulation stays high after a time shift, because the shift only moves which phase bin is tall. The recording needs about 60 s of accepted signal; otherwise the page says so and does not invent an index. If the amplitude band contains a harmonic of the phase band, the index is skipped only when the current cycles are far from symmetric: rise/decay or peak/trough at least 0.15 from one half. Otherwise the index is still shown, and the caption says so. **Map** draws a coarse grid of the index and states how many cells were tested. Surrogates are for the selected bands only, and the map is not corrected across cells.

**Quality.** Stretches the rules flag, and a way to leave them out of later fits. The file on disk is not rewritten. Each kind has its own stripe, and the legend names the kinds on the page. Dragging across a channel on the timeline adds a confirmed span for that channel. **Put back** removes it. The file is not rewritten.

| Kind | What it means |
| --- | --- |
| Clip, flat, step, amplitude | Detected. The rule is sure. |
| Muscle, drift, mains | Suspected. Listed, and left in the fit until you choose otherwise. |
| File mark, missing | Already on the file, or non-finite samples. |
| Left out | A stretch you confirmed on this page. |
| Drawn | A span dragged on this page. It is already left out. |

**Leave out** copies the recording with that stretch removed and refits. The channel you were on stays selected. **Put back** restores a stretch that was left out here. Detected and suspected rows do nothing until Leave out.

Confirmed exclusions on either split eye-state recording also apply to the original continuous recording used for event reactivity. Their source offsets are preserved. Restoring an exclusion restores those event samples as well. Live band changes reach the Time views immediately. Cached and in-progress Time jobs keep separate request identities, so a completed job cannot replace a different selected view.

## Saving

The save buttons follow the current tab and its available results. Pending or failed Time jobs have no data to export. Export results becomes available after background analysis finishes. Checking button availability does not construct the potentially large spectrogram CSV.

**Save image** writes the current page as a PNG at 150 dpi. The file carries a one-line caption naming the channel, window, applied reference, and unit. That line is removed from the on-screen figure after the save. A very small figure is written at a readable size and then restored. Findings is text; use Copy on that page.

**Save data** writes the current table.

- Spectrogram: one row per time-frequency bin that has a value, with the settings key, channel, time, sample index, frequency, and dB.
- Exponent and peaks: one row per window, with the settings key, sample index, exponent, offset, knee, peak, bandwidth, R², and the fraction of the window that was rejected.
- Bursts: one row per episode, with duration, cycles, amplitude, and frequency.
- Burst lengths: one summary row.
- Cycle shape and amplitude and frequency: one row per cycle, with the settings key and the sample indexes stored on the cycle.
- Quality: one row per span.
- Reactivity, when it is a single trace: time and band power.

**Export results** writes the traces, each condition’s spectrum, peaks, and metadata, the cycle tables, the comparison, findings, and `time.png` / `time.csv` when the Time page has rows. The metadata names the spectrogram window and overlap once that picture exists, and the sliding window once that fit has been run. A project is a separate file for reopening the review.

GUI sessions and CLI runs are staged before the destination is changed. A successful export replaces the BiFooof-owned files in that folder, removes obsolete peak/cycle/plot files, and keeps unrelated files. Canonical output names such as `metadata.json`, `peaks.csv`, `ec_spectrum.csv`, and `cycles.png` are reserved for the app. A failed calculation preserves the previous bundle; ordinary publishing errors roll it back. `bifooof-manifest.json` is written last and lists the completed output files.

Metadata uses the settings snapshot attached to the fit. It reports the requested and applied references, source offset, contributing epochs, per-channel accepted sample counts/durations, and accepted/excluded intervals on the original source clock. The top-level accepted duration is time with at least one channel contributing samples. Short runs that cannot enter Welch and incomplete epoch tails are excluded from those counts.

Coupling results identify both the phase channel and the amplitude channel. Every map cell uses those same channel choices and the same filtering/edge handling as the selected-band calculation. The map's band widths differ from the manually selected band, so their numerical indices need not match unless the band edges match.

## Saved projects and recovery

File → Save project writes a versioned `.bifooof.json` review; Open project restores
it. The file references source recordings rather than copying their sample data.
Keep those sources and companion files. SHA-256 identities cover the primary
source, BrainVision data/marker files, JSON manifest CSV files and declared EEGLAB
FDT files. Moved sources can be relinked during reopening; their contents must
match. Changed sources are rejected before restoration and can be opened as a
new recording. Event review copies follow a verified relink.

The project preserves condition assignments, source offsets, manual exclusions,
bad-channel decisions, bookmarks, notes and undo/redo history, regions, edited
events, applied settings, staged edits, selected channel and view controls.
Completed measurement summaries support the session index. Reopening recomputes
the analyses; an unfinished calculation is not restored as a completed result.

Review edits and view changes trigger a debounced recovery copy. File → Recover
last workspace reopens it through the same source verification. Atomic writes
preserve a `.previous` copy. Preferences, recovery and profile files live under
`%USERPROFILE%\.bifooof`; `BIFOOOF_STATE_DIR` selects another location. Recent
projects/folders and window geometry persist there.

## Review journal and quality overview

Review → Quality lists per-channel accepted duration, excluded percentage,
Welch support, contributing epochs, fit error/R², peak-width flags, requested
and applied reference, failure reasons and reviewer status. Comparison checks
show mismatched channel sets, reference contributors, frequency coverage,
settings, source units and clean-data support. Equal elapsed windows can contain
different amounts of usable signal. These checks are evidence to inspect;
Reviewed is a reviewer decision separate from the calculated flags.

Double-click a quality row to inspect the channel's spectrum. Mark reviewed,
exclude channel or restore it using the explicit actions. Trace visibility
does not control analysis eligibility. Original file bad channels and masks
remain excluded when a manual decision is undone.

Notes and bookmarks supports an alias, session date, observations, and bookmarks
with condition, channel scope, time, duration, label and note. Bookmarks are
informational until explicitly excluded. Their navigation opens the signal;
editing, deleting and excluding a bookmark participates in review undo/redo.

Import inspection shows original/normalized names, retained channels, analysis
eligibility, types, units, rate and known/unknown stored reference. File → Import
channel selection changes future imports between the standard 19 preference
and all remaining voltage channels. EOG/ECG/EMG and ear references can be retained
for trace review but do not enter neural fits or the EEG average reference.
MNE status/stimulation channels are omitted. Nonstandard neural channels can be
fitted, but head maps still require known electrode positions. CSV defaults to
microvolts; JSON can declare units, reference and a channel-name-to-type mapping.

## Profiles and help

Profiles saves, previews, imports and exports versioned named processing settings
including custom burst thresholds and Time controls. The Review label shows a
modified profile when controls differ. Exclusions and source identity belong to
the project. A CLI run can use the same fit snapshot with
`python -m bifooof run FILE --profile PROFILE.json`. Time controls apply in the GUI.
Help → Measurement guide defines the measures; Help → Open synthetic demonstration
creates generated EC/EO rhythms for trying the review workflow.

## Study views

- **Epoch stability:** examine retained individual epochs or compute explicit
  nonoverlapping windows with spectral and cycle summaries. Window rows link to
  the signal. The plot shows the selected channel's measurements, median and IQR;
  the summary table includes total/peak-present window counts and first/second
  half medians. These describe variability, not confidence intervals.
- **Alpha candidates:** choose a declared region and search range. Rows show
  supported fitted peaks, resolution, fit quality, epoch support and width flags.
  Accept a selected row with an explicit half-width to apply one shared band to
  both conditions. Its defining condition/channel/search is saved as band origin.
  Unsupported peaks remain missing; the band must fit the actual spectral range
  and both Nyquist limits.
- **Regions:** add/edit named channel groups. Summaries use the same available,
  usable members across conditions and list missing members. They summarize
  existing channel fits; they are not a new fit to an averaged regional PSD.
  Left/right asymmetry is `log10(mean left band PSD / mean right band PSD)`.
  Band PSD is mean linear density in µV²/Hz. Nonpositive or missing power yields
  no asymmetry value.
- **Events:** inspect and navigate source-clock events, edit review copies, map
  eye states or leave an event informational, and add/remove review events.
  EC→EO and EO→EC remain separate. Time → Reactivity uses the selected direction
  on a real continuous source, displays baseline/response windows and contributing
  event counts, and exports per-event aligned band-power rows. Partial curves
  remain visible; events lacking baseline or response are excluded from the
  direction-specific change estimate. Separate recordings retain separate courses.
- **Repeated sessions:** add saved project files to the local index, select a
  channel, measure and baseline. Dated EC/EO points share the session position.
  Baseline delta means session minus baseline; tables preserve its project,
  definition, processing settings, reference, profile, notes, accepted support
  and fit quality. Unknown/different stored references and changed processing
  are flagged. Double-click a point or table row to reopen its verified project.
  Save a project after calculations finish to update its completed summaries.

Study tables can be exported with Save data. Saved projects retain the underlying
review and source references; exporting a table does not replace that project.

## Reports, batch work and longer recordings

File → Preview report and results lets you choose sections, individual current
plots and whether source paths/filenames are included. The preview shows labels
and reviewer text before export. The offline HTML and A4 PDF contain observations,
calculated findings, quality/support, applied settings, versions and measurement
definitions from one completed snapshot. Selected plots render at a readable
size even when their tab is hidden. Reviewer notes/aliases are included as typed.

The results bundle adds versioned `review.json`, `events.tsv`/`events.json`,
`channels.tsv`, `measurement_dictionary.json`, `quality.csv`, `regions.csv` and
`epochs.csv` to the existing outputs. Event TSV times use sample zero of the
original source. These documented sidecars do not constitute a complete BIDS
dataset. Reports and sidecars appear in the export manifest.

Tools → Batch queue locks the applied processing settings for a queue. Assign
each input to EC or EO, optionally select a channel region for cycles, and choose
the output folder. Jobs run sequentially in unique destinations. A malformed
input does not stop other items. Retry failed/pending items preserves successful
outputs and aggregate rows; JSON records the locked fit settings. Cancellation
leaves completed items intact and never marks an unfinished item complete.

Background work reports its current channel/window/run where supported. Esc or
Tools → Cancel requests cooperative cancellation; a library call already running
finishes before the next checkpoint. Each request scope keeps one active and
one latest pending job. Stale results cannot replace newer requests. Time's
result cache is bounded to 128 MiB and Tools can clear it. Review masks share
source sample storage. Spectrogram/sliding CSV data streams in a worker; cancelled
exports preserve the previous destination. Skipped cycle runs retain their
reasons in caption details and cycle-summary exports.

The recorded performance fixture uses a generated one-hour, 19-channel, 256 Hz
recording: 133.6 MiB of samples, 2.39 s for whole-window spectra and 7.75 s for
one channel's cycles on this machine. Peak process working set was 639 MiB.
It includes cold library imports and excludes file decoding and GUI work;
it is one reproducible local measurement, not a bound for every recording.
