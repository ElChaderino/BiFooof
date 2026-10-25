"""Display-only filtering; original samples and analysis settings are untouched."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from .segments import accepted_runs


def display_filter(values, sfreq, band="Off", notch=0):
    """Filter a display copy.

    A stretch shorter than the filter can use stays as recorded. When every
    stretch is that short, this raises instead of returning a blank trace.
    The second value is a note for the mixed case.
    """
    result = np.array(values, dtype=float, copy=True)
    sos = None
    if band != "Off":
        low, high = map(float, band.replace("–", "-").replace("—", "-").split("-"))
        if not 0 < low < high < sfreq / 2:
            raise ValueError("Display band must lie below the recording's Nyquist frequency.")
        sos = butter(3, [low, high], fs=sfreq, btype="bandpass", output="sos")
    if notch and not 0 < notch < sfreq / 2:
        raise ValueError("Display notch must lie below the Nyquist frequency.")
    requested = sos is not None or bool(notch)
    filtered = False
    skipped = False
    for row in result:
        for start, stop in accepted_runs(np.isfinite(row)):
            if not requested:
                continue
            try:
                piece = row[start:stop]
                if sos is not None:
                    piece = sosfiltfilt(sos, piece)
                if notch:
                    b, a = iirnotch(notch, 30, fs=sfreq)
                    piece = filtfilt(b, a, piece)
                row[start:stop] = piece
                filtered = True
            except ValueError:
                skipped = True
    if requested and skipped and not filtered:
        raise ValueError(
            "This stretch is shorter than the display filter can use. "
            "Turn the filter off to see the samples."
        )
    note = "A stretch shorter than this filter was left unfiltered." if skipped and filtered else ""
    return result, note
