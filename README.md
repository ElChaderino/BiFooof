# BiFooof

Windows app for specparam (FOOOF) and bycycle on EEG.

Open an EDF, WinEEG `.eeg`, BrainVision, EEGLAB, FIF, CNT, CSV, or JSON file. BXS files need `bxs2edf.exe` on the machine. Load eyes-closed and eyes-open together and the comparison view puts the two spectra side by side.

specparam fits a chosen stretch of the recording, 1–40 Hz by default, and reports the aperiodic slope and the peaks. The fit bar can switch that to the published 2–40 Hz resting range, a narrower 3–30 or 2–20 Hz band, a 2, 4, or 8 second Welch segment, and a fixed slope or a knee. Peak width, count, height, and the 2 SD peak threshold are on that bar too. bycycle takes the channel you pick and finds the waves in one band. Alpha, 8–12 Hz, is the default, and both the cycle band and the alpha range used for reactivity can be typed in. Burst detection can stay on the library rule or use the resting-alpha and paper-alpha thresholds. The trace and cycle pages scroll. A Refit uses the window you set; the scrollbar does not change the fit.

## Screenshots

These are a 60 second section of an eyes-closed and eyes-open EDF, from 120 s to 180 s. The names in the title bar are placeholders. The trace page is 158–173 s, where the occipital 10 Hz rhythm is strongest. Nothing identifying was left in the window.

![Traces](docs/screenshots/traces.png)

![Spectrum](docs/screenshots/spectrum.png)

![Cycles](docs/screenshots/cycles.png)

![Compare](docs/screenshots/compare.png)

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

## One file

```text
pip install pyinstaller
pyinstaller bifooof.spec
```

The result is `dist\BiFooof.exe`. Copy that file by itself. The first launch unpacks into a temporary folder, so the window takes a little longer to appear.
