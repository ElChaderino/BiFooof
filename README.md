# BiFooof

Windows app for specparam (FOOOF) and bycycle on EEG.

Open an EDF, WinEEG `.eeg`, BrainVision, EEGLAB, FIF, CNT, CSV, or JSON file. The `.eeg` reader is part of the app. BXS files need `bxs2edf.exe` on the machine. Open an eyes-closed recording and an eyes-open recording together and the comparison view puts the two spectra side by side.

## Fit bar

specparam fits a stretch of the recording and reports the aperiodic slope and the peaks. The default is the whole file, 1–40 Hz, a 4 second Welch segment (0.25 Hz bins), and a fixed slope.

The fit bar can switch that to the published 2–40 Hz eyes-open / eyes-closed range, a narrower 3–30 or 2–20 Hz band, a 2 or 8 second segment, and a knee instead of a fixed slope. Peak width, count, height, and the 2 SD peak threshold are on that bar too.

**Refit** runs specparam and bycycle on the time window you set. The scrollbars on the trace and cycle pages only move the picture.

## Band

Alpha, 8–12 Hz, is the default. The same control is delta 1–4, theta 4–8, beta 13–30, gamma 30–45, or any range you type. The Cycles page and the eyes-closed versus eyes-open reactivity both follow it. Changing the band does not refit the spectrum.

Burst detection can stay on the library rule, or use the resting-alpha and paper-alpha amplitude rules. Those rules still apply to whichever band is selected.

## Drift, bursts, and marks

The spectrum fit removes a straight drift line from each channel before the Welch estimate. The cycle fit does the same before its bandpass.

When cycles are found and none of them form a burst, the Cycles page names the cutoff that stopped the run. The library rule needs monotonicity above 0.8 for three cycles in a row.

A WinEEG mark that says eyes closed or eyes open cuts that side out of the file. A file with both marks opens both sides. A clock-time mark is kept as a note and does not cut the recording. Two separate files still open exactly as before.

Samples marked `BAD` in an MNE file, bad channels, and nonfinite samples stay out of the spectrum and the cycle fit. They are not replaced with zeros, and a cycle cannot cross a rejected gap.

The comparison has two different numbers. Peak Δ is eyes-closed fitted peak height minus eyes-open fitted peak height, above the aperiodic background, in log10 power. Each side picks its own peak, and a missing peak stays missing. Δ band % is the percent change between the two recordings in mean Welch power inside that band, `(EC − EO) / EO × 100`. It is not the band's share of total spectral power. It stays blank when eyes-open power is missing or not positive. The spectrum table's Height column is that same fitted peak height, not Welch band power.

Epochs, 30 s or 60 s, are non-overlapping and each one is fit on its own. Peaks whose centers fall within 1 Hz are one group. The reported center, height, and width are the medians of the epochs in that group. An epoch that missed the peak is left out of the median. It is not entered as a height of zero, so a peak that appears less often does not look shorter. The caption counts how many epochs contained an in-band peak, and that count does not scale the height. Offset, exponent, and R² are medians across epochs that fit. The spectrum curve is their mean. Press Refit after changing epochs, the reference, or the amplitude limit.

A weak fit, R² below 0.6, is ringed on the head map. A peak at the narrowest allowed width is marked. A file whose voltage looks like volts, or far above a normal EEG range, is labelled on the file name and is not rescaled. The reference in use is named on the spectrum and comparison captions. The default is the reference stored in the file.

A burst count is split in two. Burst cycles are cycles that passed the burst rule. Burst episodes are the separate runs of those cycles. Save session writes `metadata.json` and a one-row-per-peak table beside the spectrum CSV.

## Time

The Time tab uses the same fit window, reference, and rejected samples as the spectrum. Spectrogram is the default: power across the recording, with burst episodes drawn on it when the cycle fit has them. Click a time and the trace scrolls there. Set fit window writes that stretch into the fit bar; Refit still has to be pressed.

The other views on that tab are the sliding specparam fit, burst episodes, burst lengths, cycle shape, amplitude against cycle frequency, band power around eye marks when a file actually has them, coherence, phase–amplitude coupling, and a quality timeline. Two separate files are not given an invented transition. Coupling is off the ordinary refit and says when the recording is too short.

What each page and control does, and what is not in the window yet, is in [docs/running.md](docs/running.md). That file is the one to change when the window changes.

## Screenshots

These are a 60 second section of an eyes-closed and eyes-open EDF, from 120 s to 180 s. The names in the title bar are placeholders. The trace page is 158–173 s, where the occipital 10 Hz rhythm is strongest. The Time page is the spectrogram of that same window. Findings is the reading of this fit. Nothing identifying was left in the window.

![Traces](docs/screenshots/traces.png)

![Spectrum](docs/screenshots/spectrum.png)

![Time](docs/screenshots/time.png)

![Cycles](docs/screenshots/cycles.png)

![Compare](docs/screenshots/compare.png)

![Findings](docs/screenshots/findings.png)

## Run from source

```text
pip install -r requirements.txt
python -m bifooof
```

From a command prompt:

```text
python -m bifooof info FILE
python -m bifooof run FILE --condition ec --channel O1 --out bifooof-out
python -m bifooof compare --ec EC_FILE --eo EO_FILE --channel O1 --out bifooof-out
python -m bifooof formats
```

`--band` sets the cycle band: `delta`, `theta`, `alpha`, `beta`, `gamma`, or a range such as `8-12`. `--alpha LOW HIGH` sets the reactivity peak. In the window those two stay on the same band.

## One file

```text
pip install pyinstaller
pyinstaller bifooof.spec
```

The result is `dist\BiFooof.exe`. Copy that file by itself. The first launch unpacks into a temporary folder, so the window takes a little longer to appear.
