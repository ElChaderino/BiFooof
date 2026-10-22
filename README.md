# BiFooof

Windows app for specparam (FOOOF) and bycycle on EEG.

Open an EDF, WinEEG `.eeg`, BrainVision, EEGLAB, FIF, CNT, CSV, or JSON file. BXS files need `bxs2edf.exe` on the machine. Load eyes-closed and eyes-open together and the comparison view puts the two spectra side by side.

specparam fits the whole recording from 1–40 Hz and reports the aperiodic slope and the peaks. bycycle takes the channel you pick and finds the waves in one band. Alpha, 8–12 Hz, is the default. The trace and cycle pages scroll. The fits use the whole file.

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
