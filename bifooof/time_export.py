"""Streaming rows from a frozen Time-result reference, without Qt access."""
import numpy as np
from .analysis.timefreq import Spectrogram, SlidingFit, channel_series
from .jobs import checkpoint


def time_rows(snapshot):
    result, rate, key = snapshot["result"], snapshot["rate"], snapshot["key"]
    if isinstance(result, Spectrogram):
        for index, moment in enumerate(result.times_s):
            checkpoint(index, len(result.times_s), f"Writing spectrogram column {index + 1}/{len(result.times_s)}")
            for freq, level in zip(result.freqs_hz, result.power_db[:, index]):
                if np.isfinite(level):
                    yield {"settings_key": key, "channel": result.channel, "time_s": round(float(moment), 3),
                        "sample": round(float(moment) * rate), "freq_hz": round(float(freq), 3), "power_db": round(float(level), 3)}
    elif isinstance(result, SlidingFit):
        series = channel_series(result, snapshot["channel"], snapshot["band"])
        def number(value):
            return None if value is None or not np.isfinite(value) else float(value)
        for index, moment in enumerate(series.times_s):
            checkpoint(index, len(series.times_s), "Writing sliding windows")
            yield {"settings_key": key, "channel": snapshot["channel"], "time_s": round(float(moment), 3),
                "sample": round(float(moment) * rate), "exponent": number(series.exponent[index]), "offset": number(series.offset[index]),
                "knee": number(None if series.knee is None else series.knee[index]), "peak_hz": number(series.peak_hz[index]),
                "peak_height": number(series.peak_height[index]), "bandwidth_hz": number(series.peak_bandwidth[index]),
                "r_squared": number(series.r_squared[index]), "artifact_fraction": number(series.artifact_fraction[index])}
    else:
        yield from snapshot.get("rows", [])
