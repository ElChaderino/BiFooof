"""Display-only filtering; original samples and analysis settings are untouched."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from .segments import accepted_runs


def display_filter(values, sfreq, band="Off", notch=0):
    result = np.array(values, dtype=float, copy=True)
    sos = None
    if band != "Off":
        low, high = map(float, band.split("–"))
        if not 0 < low < high < sfreq / 2:
            raise ValueError("Display band must lie below the recording's Nyquist frequency.")
        sos = butter(3, [low, high], fs=sfreq, btype="bandpass", output="sos")
    if notch and not 0 < notch < sfreq / 2:
        raise ValueError("Display notch must lie below the Nyquist frequency.")
    for row in result:
        for start, stop in accepted_runs(np.isfinite(row)):
            if sos is None and not notch:
                continue
            try:
                piece = row[start:stop]
                if sos is not None:
                    piece = sosfiltfilt(sos, piece)
                if notch:
                    b, a = iirnotch(notch, 30, fs=sfreq)
                    piece = filtfilt(b, a, piece)
                row[start:stop] = piece
            except ValueError:
                row[start:stop] = np.nan
    return result
