"""Read a WinEEG legacy ``.eeg`` file into physical sample values.

This is the read side of the Mitsar converter. It lives in the app so a
clean install can open ``.eeg`` files without a separate checkout.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LEGACY_MAGIC = 0x0F0E
LEGACY_HEADER_SIZE = 0x80
LEGACY_CHANNEL_DESCRIPTOR_SIZE = 0x40

HEADER_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("magic", 0x00, "H"),
    ("channel_count", 0x02, "H"),
    ("sample_rate_hz", 0x04, "H"),
    ("primary_mark_count", 0x06, "H"),
    ("text_blob_size", 0x08, "H"),
    ("sample_width_bytes", 0x0A, "H"),
    ("bits_per_sample", 0x0C, "H"),
    ("extended_mark_count", 0x0E, "H"),
    ("logical_frame_count", 0x10, "I"),
    ("profile_offset", 0x14, "I"),
    ("channel_table_offset", 0x18, "I"),
    ("sample_data_offset", 0x1C, "I"),
    ("primary_marks_offset", 0x20, "I"),
    ("extended_marks_offset", 0x24, "I"),
    ("text_blob_offset", 0x28, "I"),
    ("word_0x2c", 0x2C, "H"),
    ("word_0x2e", 0x2E, "H"),
    ("word_0x30", 0x30, "H"),
    ("index_count", 0x32, "H"),
    ("index_offset", 0x34, "I"),
    ("text_line_count", 0x38, "H"),
    ("montage_offset", 0x3A, "I"),
    ("format_flags", 0x3E, "H"),
    ("alternate_block_count", 0x48, "H"),
    ("alternate_block_offset", 0x4A, "I"),
    ("alternate_block_frames", 0x4E, "H"),
    ("event_map_count", 0x50, "H"),
    ("event_group_count", 0x52, "H"),
    ("event_map_offset", 0x54, "I"),
    ("event_group_offset", 0x58, "I"),
    ("settings_offset", 0x5C, "I"),
    ("aux_record_offset", 0x60, "I"),
    ("aux_record_count", 0x64, "H"),
    ("float_table_present", 0x66, "H"),
    ("float_table_offset", 0x68, "I"),
    ("view_descriptor_count", 0x6C, "H"),
    ("view_descriptor_offset", 0x6E, "I"),
    ("extra_descriptor_count", 0x72, "H"),
    ("extra_descriptor_offset", 0x74, "I"),
    ("optional_block_offset", 0x78, "I"),
    ("optional_block_size", 0x7C, "H"),
)


@dataclass(frozen=True)
class LegacyChannel:
    index: int
    label: str
    source_code: int
    reference_code: int
    field_0x0c: int
    field_0x0e: int
    field_0x10: int
    x: int
    y: int
    scale: float
    kind: int
    correction: float
    zero_offset: float


@dataclass
class LegacyEEG:
    header: dict[str, int]
    channels: list[LegacyChannel]
    raw: bytes
    physical_frame_count: int
    trailing_sample_bytes: int


def _unpack(raw: bytes, offset: int, fmt: str) -> int:
    return int(struct.unpack_from("<" + fmt, raw, offset)[0])


def _decode_fixed(raw: bytes, *, encoding: str = "cp1252") -> str:
    value = raw.split(b"\0", 1)[0].rstrip(b" ")
    return value.decode(encoding, errors="replace")


def parse_legacy_header(raw: bytes) -> dict[str, int]:
    if len(raw) < LEGACY_HEADER_SIZE:
        raise ValueError("legacy EEG is shorter than the 128-byte header")
    values = {name: _unpack(raw, offset, fmt) for name, offset, fmt in HEADER_FIELDS}
    if values["magic"] != LEGACY_MAGIC:
        raise ValueError(f"unexpected legacy EEG magic 0x{values['magic']:04x}")
    if not 1 <= values["channel_count"] <= 256:
        raise ValueError("legacy EEG channel count is outside 1..256")
    if not 1 <= values["sample_rate_hz"] <= 20000:
        raise ValueError("legacy EEG sample rate is outside 1..20000 Hz")
    if values["sample_width_bytes"] not in {2, 3}:
        raise ValueError("only the recovered 16-bit and 24-bit legacy sample widths are supported")
    return values


def parse_legacy_channels(raw: bytes, header: dict[str, int]) -> list[LegacyChannel]:
    start = header["channel_table_offset"]
    end = start + header["channel_count"] * LEGACY_CHANNEL_DESCRIPTOR_SIZE
    if start < LEGACY_HEADER_SIZE or end > len(raw):
        raise ValueError("legacy EEG channel table is outside the file")
    channels: list[LegacyChannel] = []
    for index in range(header["channel_count"]):
        offset = start + index * LEGACY_CHANNEL_DESCRIPTOR_SIZE
        channels.append(
            LegacyChannel(
                index=index,
                label=_decode_fixed(raw[offset : offset + 8], encoding="ascii") or f"CH{index + 1}",
                source_code=_unpack(raw, offset + 0x08, "h"),
                reference_code=_unpack(raw, offset + 0x0A, "h"),
                field_0x0c=_unpack(raw, offset + 0x0C, "h"),
                field_0x0e=_unpack(raw, offset + 0x0E, "h"),
                field_0x10=_unpack(raw, offset + 0x10, "h"),
                x=_unpack(raw, offset + 0x14, "h"),
                y=_unpack(raw, offset + 0x16, "h"),
                scale=float(struct.unpack_from("<f", raw, offset + 0x18)[0]),
                kind=_unpack(raw, offset + 0x1C, "h"),
                correction=float(struct.unpack_from("<f", raw, offset + 0x26)[0]),
                zero_offset=float(struct.unpack_from("<f", raw, offset + 0x2A)[0]),
            )
        )
    return channels


def parse_legacy_eeg(raw: bytes) -> LegacyEEG:
    header = parse_legacy_header(raw)
    channels = parse_legacy_channels(raw, header)
    start = header["sample_data_offset"]
    end = header["primary_marks_offset"]
    if not (0 <= start <= end <= len(raw)):
        raise ValueError("legacy EEG sample interval is not inside the file")
    frame_bytes = header["channel_count"] * header["sample_width_bytes"]
    sample_bytes = end - start
    return LegacyEEG(
        header=header,
        channels=channels,
        raw=raw,
        physical_frame_count=sample_bytes // frame_bytes,
        trailing_sample_bytes=sample_bytes % frame_bytes,
    )


def read_legacy_eeg(path: Path) -> LegacyEEG:
    return parse_legacy_eeg(path.read_bytes())


def _legacy_digital_frames(eeg: LegacyEEG) -> Iterable[tuple[int, ...]]:
    header = eeg.header
    start = header["sample_data_offset"]
    channels = header["channel_count"]
    width = header["sample_width_bytes"]
    if width == 2:
        fmt = "<" + "h" * channels
        stride = channels * 2
        for offset in range(start, start + eeg.physical_frame_count * stride, stride):
            yield struct.unpack_from(fmt, eeg.raw, offset)
        return
    stride = channels * 3
    for offset in range(start, start + eeg.physical_frame_count * stride, stride):
        frame = []
        for channel in range(channels):
            pos = offset + channel * 3
            value = int.from_bytes(eeg.raw[pos : pos + 3], "little", signed=True)
            frame.append(value)
        yield tuple(frame)


def legacy_digital_channels(eeg: LegacyEEG, *, logical_only: bool = True) -> list[list[int]]:
    frame_count = eeg.physical_frame_count
    if logical_only:
        frame_count = min(frame_count, eeg.header["logical_frame_count"])
    result = [[] for _ in eeg.channels]
    for frame_index, frame in enumerate(_legacy_digital_frames(eeg)):
        if frame_index >= frame_count:
            break
        for channel, value in enumerate(frame):
            result[channel].append(value)
    return result


def legacy_physical_channels(eeg: LegacyEEG, *, logical_only: bool = True) -> list[list[float]]:
    digital = legacy_digital_channels(eeg, logical_only=logical_only)
    result: list[list[float]] = []
    for descriptor, values in zip(eeg.channels, digital):
        scale = descriptor.scale if math.isfinite(descriptor.scale) and descriptor.scale != 0 else 1.0
        zero = descriptor.zero_offset if math.isfinite(descriptor.zero_offset) else 0.0
        result.append([(value - zero) * scale for value in values])
    return result
