# BiFooof 0.2.0 additions and original specification

Implemented on 2026-10-23, prioritizing individual EEG review and
eyes-closed/eyes-open comparison. All fourteen approved additions below now
have working implementation paths in 0.2.0. The original requirements and
baseline observations are retained after the delivery index for traceability.
The running guide describes the actual menus, clocks, formulas and limitations.

## Delivery index

- **F01:** verified project save/open/relink, recent projects, atomic previous
  copies, recovery, applied/staged settings and view restoration.
- **F02:** shared review journal, notes, bookmarks, manual channel/interval
  exclusions, navigation and persisted undo/redo that preserve source masks.
- **F03:** per-channel signal support, review status, model flags, reference
  contributors/fallbacks and EC/EO comparability checks.
- **F04:** named versioned profiles, GUI/CLI fit-setting equality, preview,
  measurement guide, generated demonstration and saved preferences.
- **F05:** section/plot/path preview, offline HTML and visually checked A4 PDF,
  generated from a completed snapshot and listed in the results manifest.
- **F06:** extrema/gap-preserving traces and cycles, paging, jump, interval/ruler,
  overlays, display filters and supported display montages.
- **F07:** retained epoch rows, nonoverlapping spectral/burst stability,
  median/IQR, peak-presence counts and first/second-half summaries.
- **F08:** supported regional alpha candidates with search/resolution/quality,
  explicit acceptance, shared comparison band and saved band origin.
- **F09:** saved/custom regions, common usable members, missing-channel details,
  channel-parameter summaries and explicitly defined log-power asymmetry.
- **F10:** editable review events, state mapping/navigation, separate transition
  directions, support counts/windows and aligned event exports.
- **F11:** cooperative cancellation/progress, bounded pending requests and LRU
  cache, shared source buffers, streamed exports and cycle-run failure details.
- **F12:** local saved-project session index, channel/measure/baseline selection,
  dated trends, comparison flags, notes/profile/settings and evidence reopening.
- **F13:** sequential GUI queue with explicit conditions, locked settings,
  channel groups, isolated/unique outputs, cancellation and preserved retry totals.
- **F14:** import/normalization/type/eligibility inspection, configurable retention,
  auxiliary/status handling, versioned review/event/channel/measure sidecars.

Verification includes 125 passing tests (the previous 95 plus 30 workflow
regressions), nine real Qt workflow checks, minimum-window screenshots and
rendered PDF checks. A generated one-hour/19-channel/256-Hz fixture records
133.6 MiB sample storage, 2.39 s spectra, 7.75 s single-channel cycles and
639 MiB peak working set on this machine. See `runs/roadmap` and
`runs/code-walk/evidence.json` for reproducible local evidence. Packaged-build
checks and SHA-256 evidence are written after the Windows build.

Projects reference recordings and recompute analysis when reopened. The session
index reads completed saved summaries. Cancellation takes effect between library
calls. Head maps still require supported electrode positions, and the sidecars
are not a complete BIDS dataset. The separately scoped interpretation, ICA,
localization and large-connectivity projects below remain outside this release.

## Original 0.1.5 proposal

The highest-value next work is preserving a review, making the supporting
signal and comparison conditions visible, and producing a useful report.
The application already has spectra, cycle features, time views, coherence,
PAC, quality proposals, and findings. The following additions build on those
capabilities.

## Baseline observations before implementation

- `MainWindow.save_session()` and `session_export.save_bundle()` export a
  completed collection of images, tables, findings, and metadata. There is no
  project-open path that restores the workspace.
- `TimePanel.excluded` holds review exclusions in memory. Quality has Leave
  out and Put back, but there is no general review journal or undo history.
- `TracePanel` supports channel visibility, selection, page length, gain, and
  scrolling. Unchecking a channel hides its trace; it does not exclude that
  channel from analysis.
- `FitSettings` and FitBar provide built-in presets. User-defined, named
  processing profiles are not saved across launches.
- Whole-recording comparisons use the shorter elapsed duration. Effective
  contributing samples are exported per channel, but there is no consolidated
  screen explaining comparison compatibility and usable-data differences.
- Epoch fitting retains combined parameters, peak counts, and contributing
  intervals. It does not retain a browsable table of all individual epoch fits.
  Sliding analysis already has window-level results to build on.
- The CLI can process multiple files. The GUI loads two conditions and does
  not provide a multi-recording queue or a longitudinal session library.
- Workers discard stale results correctly. They have no progress callback or
  cancellation contract, and the Time result cache has no explicit size limit.

These observations come from `ui/main_window.py`, `ui/time_panel.py`,
`ui/trace_panel.py`, `analysis/settings.py`, `analysis/specparam_fit.py`,
`analysis/prepare.py`, `cli.py`, `export.py`, and `session_export.py`.

## First additions: preserve and support daily review

### F01. Save and reopen a project

**User benefit:** reopen yesterday's review without finding the two files,
recreating the exclusions, and remembering the fit settings.

Add Save project, Open project, recent projects, and recovery autosaves. Keep
the current export operation available under an explicit Export results name.
A project should preserve source references, condition assignments, original
source offsets, exclusions, bad-channel choices, applied settings, staged
edits, selected channel, view controls, review notes, and selected comparisons.
Notes and review state should be saveable while analysis is running; partial
analysis results should not be recorded as completed results.

Use a versioned project schema. Identify the source and required companion
files, including BrainVision companions and a JSON manifest's CSV. Handle
moved files with a relink flow. Verify identity before restoring cached results;
a changed source must not inherit an old review silently. Reference the source
recordings by default rather than duplicating them into every project. Write
projects atomically and preserve the previous recovery copy.

**Completion check:** load a marked recording, split its conditions, exclude a
channel-specific interval, stage a different fit window, save, close, and
reopen. The review state and source clocks must match. Missing or changed source
files must be recoverable without applying stale results.

**Scope:** large foundation. Introduce a shared project/review model rather
than serializing Qt widgets. This supports most later additions.

### F02. Notes, bookmarks, exclusions, and undo

**User benefit:** record why an interval matters and move through the review
without losing decisions.

Add timestamped bookmarks with a label, channel scope, duration, and note;
next/previous bookmark navigation; and a review list linked to Traces and Time.
Support whole-channel analysis exclusion separately from trace visibility.
Keep informational annotations separate from analysis exclusions. Add undo and
redo for review edits, including restoring overlapping exclusion intervals.

Distinguish source-file exclusions from the reviewer's additions: undoing a
review edit must preserve the original file's bad-data masks. Save the review
journal with F01. Show event and exclusion overlays on the trace, with their
scope and reason available on selection.

MNE documents editable annotations and separate bad-channel tracking, which
are useful patterns for this workflow. BiFooof should preserve its existing
source-local clock contract when adopting them.
[MNE annotations](https://mne.tools/stable/auto_tutorials/raw/30_annotate_raw.html),
[MNE bad channels](https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html).

**Completion check:** an informational bookmark leaves fit inputs unchanged;
an exclusion updates the fits; undo restores the preceding mask; and all three
states survive project reopening.

**Scope:** medium to large; depends on F01's review model.

### F03. Review overview and comparison readiness

**User benefit:** answer which channels and measurements need attention before
reading an EC/EO difference.

Add an overview with per-condition and per-channel usable duration, percentage
excluded, contributing epochs or Welch segments, missing channels, requested
and applied reference, fit error, residual inspection links, width-limit flags,
and analysis failures. Flagged rows should jump to the relevant channel and
time. Reviewer status should be explicit, such as Unreviewed or Reviewed, and
remain separate from automatically calculated quality indicators.

Add comparison checks for effective channel sets, applied references and their
contributors, sampling rates and usable frequency ranges, fit settings, source
units, elapsed fit windows, and contributing clean signal. Equal elapsed
windows can contain unequal usable data after rejection. Show the difference
and allow explicit selection of suitable intervals. A shared sampling rate is
not required for every spectral comparison, so classify incompatibilities by
the measure being computed instead of blocking every difference.

Do not collapse all this into a universal reliability score. Specparam's own
evaluation guide explains that very high R² can also reflect overfitting.
Residuals, data support, peak constraints, and stability provide complementary
evidence.
[Specparam model evaluation](https://specparam-tools.github.io/auto_tutorials/plot_06-Metrics.html).

**Completion check:** equal-length synthetic recordings with different masks
show different contributing durations. A requested linked reference that fell
back to stored reference is visible. Selecting a flag opens its evidence.

**Scope:** medium. Much of the basic information already exists in fit metadata;
Welch contribution counts and reference-contributor details need additions.

### F04. Saved processing profiles and useful help

**User benefit:** repeat the same analysis without reconstructing its settings.

Allow named, versioned profiles covering fit settings, burst thresholds,
analysis band, reference, epoch choices, and relevant Time controls. Export and
import profiles, show a preview of changes, and mark a modified profile as
modified. Keep the applied profile distinct from staged edits. Profiles should
not accidentally inherit a recording's exclusions or identity.

Add short explanations beside unfamiliar measurements: fitted peak height,
mean band PSD, exponent, burst cycles, burst episodes, occupancy, and the
direction of a difference. Add a small synthetic demonstration project for
onboarding. Remember window geometry and recent folders as user preferences.

**Completion check:** applying the same profile in the GUI and CLI produces the
same settings snapshot; restoring preferences does not change an already saved
project's analysis silently.

**Scope:** small to medium; a useful early delivery alongside F01/F03.

### F05. A report assembled from the reviewed results

**User benefit:** produce one readable document instead of manually collecting
PNGs, CSV files, and findings text.

Start with an offline HTML report and a printable summary; add PDF export as a
separate, verified output. Include the recording labels, review notes, usable
signal summary, applied settings and software versions, selected plots,
condition differences, and explicit missing-result reasons. Let the reviewer
choose sections and preview identifying labels and file paths before export.
Keep reviewer observations distinct from automatically generated findings.

Generate every section from one consistent analysis snapshot. Reuse the export
staging contract, and list report outputs in the manifest. Keep the detailed
CSV/JSON bundle available for checking the measurements. MNE's report workflow
is a useful reference for assembling plots and narrative into HTML; choosing a
specific rendering library can wait until implementation.
[MNE reports](https://mne.tools/stable/auto_tutorials/intro/70_report.html).

**Completion check:** every displayed number agrees with the accompanying
table, report labels follow the chosen pair names, and a pending replacement
fit cannot mix old and new results into one report.

**Scope:** medium; most useful after F02/F03.

## Additions for examining the signal and its stability

### F06. A stronger trace review workbench

Add keyboard paging, jump to time, synchronized interval selection, a ruler
for time and amplitude, bookmark navigation, and fit-window overlays. Offer
clearly labeled display filters and display montages, with the display state
kept separate from analysis settings. Preserve the available reference and
channel information needed to construct each montage.

An immediate prerequisite is preserving brief extrema in display downsampling.
`TracePanel._render()` currently takes every Nth sample. In a synthetic 30 s,
1024 Hz trace, a 250 µV impulse at sample 12,345 disappears with the calculated
stride of 12: the original maximum is 250 µV and the displayed maximum is zero.
This demonstrates a display omission, not a change to the analysis samples.
Use an extrema-preserving display representation and retain the original
sample positions for selection and measurement.

**Completion check:** short positive and negative spikes remain visible at
every supported page length. Gap boundaries remain gaps. Moving the display
or changing its filter leaves the applied analysis snapshot unchanged.

**Scope:** medium; the downsampling correction is a focused early fix.

### F07. Epoch distributions and measurement stability

Retain individual epoch results with times, fit status, contributing data, and
peak assignments. Show distributions and median/IQR for exponent, peak center,
height, and burst measures, plus peak-presence counts. Add a first-half versus
second-half comparison and links from outlying epochs to the signal.

Label variability summaries as variability. Confidence intervals would require
a declared resampling method that accounts for temporal dependence; overlapping
sliding windows are not independent repeats. Avoid adding a significance label
to a single recording's change without defining the statistical question.

**Completion check:** stable and drifting synthetic rhythms look different in
the stability view; missing peaks remain missing; aggregate values agree with
the retained epoch table.

**Scope:** large because the current whole-window epoch path discards most
individual results after combining them.

### F08. Individual alpha frequency and reviewer-selected bands

Offer an alpha peak candidate from a declared posterior channel or region,
with its search range, resolution, fit quality, and epoch support visible.
Allow the reviewer to accept a band around that candidate or retain a fixed
band. Return No supported peak when appropriate.

For a comparison, record which condition or baseline defined the band and use
the same selected edges on both sides. Do not silently choose a different band
for each condition or change a saved project's band. Peak-width choices should
also be checked against frequency resolution; specparam documents this
relationship in its tuning guide.
[Specparam tuning](https://specparam-tools.github.io/auto_tutorials/plot_10-TroubleShooting.html).

**Completion check:** a supported peak outside the default band can be reviewed
and selected; a peak-free spectrum does not generate a made-up alpha frequency;
the accepted band and its origin survive export and reopening.

**Scope:** medium; benefits from F07.

### F09. Regions and explicit left/right summaries

Add saved channel groups such as posterior, central, and frontal regions,
alongside custom groups. Show member channels and their usable-data counts.
Use comparable channel sets across conditions and make missing members visible.

Start with summaries of the existing per-channel measurements. A spectrum fit
to an averaged regional PSD would be a different measure and should be added
under a separate definition. If asymmetry is added, name its formula, units,
frequency band, and reference. Do not mix fitted peak height with integrated
band power or log-power differences.

**Completion check:** reordering channel names does not change a region; a bad
or absent member is shown; an average of channel parameters is distinguishable
from parameters fit to a regional spectrum.

**Scope:** medium; depends on F03 and explicit channel-group definitions.

### F10. Event review and separate transition directions

Add an event list with label mapping, event navigation, editable review copies,
and source-clock export. Let the reviewer select event types and distinguish
EC→EO from EO→EC transitions. Show included/excluded event counts and the
baseline and response windows for each estimate.

The current `_reactivity()` collects onsets of both named eye states into one
event selection. Separate direction groups would make those transitions easier
to interpret and inspect. Preserve the existing rule that separately opened
files do not become an invented continuous transition.

Export events with documented onset, duration, type, and optional channel
columns. BIDS provides an established convention for these fields; supporting
these tables alone would not constitute complete BIDS dataset support.
[BIDS event specification](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html).

**Completion check:** opposite transition directions remain separate; cropping
and split-condition exclusions preserve original event alignment; unsupported
events are reported rather than silently reassigned.

**Scope:** medium to large.

## Foundations for longer recordings and larger workloads

### F11. Useful progress, cancellation, and bounded caches

Show the current task and its progress, such as channel 7 of 19 or epoch 12 of
40. Add cooperative cancellation between channels, epochs, accepted runs, and
surrogate batches. A single non-interruptible library call may have to finish
before cancellation takes effect; the UI should make that state clear.

Bound cached results by memory and evict unused entries. Avoid repeated copies
of the same recording when a compact mask revision or immutable shared data
will suffice. Stream large CSV exports in the background with progress.
Retain source/version/mask identities in every request and cached result.
Expose skipped segment counts and failure reasons in diagnostics rather than
representing every skipped cycle-analysis run as simply No cycles.

**Completion check:** rapid setting changes do not accumulate unbounded jobs
and caches; cancelling a task cannot publish partial results or replace a
different current request; long-recording memory use is measured and recorded.

**Scope:** large foundation. Start before enabling large batch workloads.

### F12. Repeated-session comparison

Add a local session index with participant aliases, session dates, conditions,
review notes, and processing-profile identity. Plot selected measures over
sessions and open the source evidence from any point. Show reference, channel,
quality, and settings changes alongside the trend; allow selecting a baseline.

Keep recorded measurement differences distinct from claims about what caused
them. Build this on saved projects, comparison checks, and stable measure
definitions so the trend can be reconstructed later.

**Completion check:** incompatible sessions are visibly identified; changing
the baseline updates the displayed direction and exported definition; each
point links to the exact saved source and analysis snapshot.

**Scope:** large; depends on F01/F03/F04, with F07 improving its usefulness.

### F13. A GUI batch queue

Build on the existing multi-file CLI. Provide explicit file-to-condition
assignment, a locked profile, optional channel groups, per-file progress,
cancellation, failure recovery, and an aggregate results table. Keep file
failures isolated. Use source identities or unique run identifiers so two
recordings with the same basename do not share an output destination.

**Completion check:** a malformed input does not stop the other files; a
cancelled item is not marked complete; rerunning only failed files preserves
successful outputs and records which profile produced them.

**Scope:** large; depends on F04/F11 and benefits from F01/F03.

### F14. Import inspection and interoperable review exports

Add an import summary showing original and normalized channel names, included
and set-aside channels, units, sampling rate, auxiliary signals, and known or
unknown reference information. The loader currently prefers the standard 19
when available and sets other channels aside. Make that choice visible and
eventually configurable, especially for EOG/ECG review and denser montages.

Add structured review-annotation exports, a dictionary of measurement names
and units, and a clear schema version. Later, consider BIDS-compatible channel
and event sidecars. Preserve original labels and available channel types before
normalization; event/status channels need special handling rather than being
mistaken for voltage traces.

BIDS EEG metadata distinguishes channel descriptions, electrode positions,
units, and recording/reference information, providing a useful target for this
work.
[BIDS EEG specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html).

**Completion check:** an import with auxiliary and nonstandard EEG channels
explains what was retained; unknown reference remains unknown; exported review
times and channel mappings round-trip without changing their meaning.

**Scope:** medium to large; capture original metadata early even if broader
montage support arrives later.

## Original implementation sequence

1. Correct the extrema-losing trace display from F06. Extract a shared
   project/review state for F01. Deliver save/open/relink/recovery before
   expanding stored analysis types.
2. Add F04 profiles and F03's basic usable-data and comparison summary. These
   are useful independently and make saved projects easier to interpret.
3. Add F02 notes/bookmarks/undo and channel exclusions, then F05 reports built
   from the same reviewed snapshot.
4. Develop F11 cancellation, progress, and cache limits alongside that work.
   Measure long-file behavior before expanding batch or all-channel jobs.
5. Add F07 stability, F08 individual bands, F09 regional summaries, and F10
   event-direction review in separate, testable changes.
6. Choose F12 longitudinal review or F13 batch processing according to actual
   user demand. Grow F14 interoperability with real input/export fixtures.

For individual reviewers, the first useful feature release is F01 + the basic
F03/F04 workflow. Reporting and the review journal should follow closely. For a
research-oriented release, move F11 and F13 earlier after those foundations.
These are dependency and value judgments, not delivery-date estimates.

## Features to scope as separate projects

Normative z-score interpretation needs a suitable reference dataset, matched
processing definitions, and a validation plan. Automated diagnoses or training
protocol recommendations would require a substantially different product and
evidence base. Source localization needs geometry and head-model support that
the current standard-19 display does not provide. Automatic ICA correction and
large connectivity matrices also require additional review and validation
workflows. They should have explicit specifications before implementation.

## Original validation plan

Keep the existing 95 tests as the baseline. Add project round-trip and recovery
cases; source relink/change cases; exclusion/undo and source-mask preservation;
profile equality across GUI/CLI; comparisons with unequal usable signal; and
report-to-table consistency. Include synthetic no-peak, drifting-peak, bad-channel,
and opposite-transition recordings. Add visual checks at the minimum window
size and a long-file performance fixture. Run reader and frozen-EXE checks when
an addition changes loading or packaging.

The library links above were consulted for workflow and measurement guidance.
They are not a request to upgrade the project's tested dependencies.
