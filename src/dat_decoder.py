"""Low-level Prophesee DAT decoding and time-window readers.

This module contains only file I/O and event decoding.  It deliberately knows
nothing about RGB timing, ROIs or colour reconstruction, which makes it useful
for testing the raw event stream independently from the rest of the pipeline.

The current parser follows the format used by the original project:
    - CD DAT version 2
    - event type 0 or 12
    - 8-byte events (uint32 timestamp + uint32 packed address/polarity)
    - x: bits 0..13, y: bits 14..27, polarity: bit 28
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DatHeader:
    event_size: int
    data_start: int
    file_first_timestamp: int
    width: Optional[int]
    height: Optional[int]
    event_type: int
    header_lines: tuple[str, ...]


def read_dat_header(file_path: str | Path) -> DatHeader:
    """Read and validate the header of a Prophesee-style DAT file."""
    file_path = Path(file_path)
    header_lines: list[str] = []
    width = None
    height = None

    with file_path.open("rb") as file:
        while True:
            first = file.read(1)
            if not first:
                raise ValueError("The file is empty or no event data were found.")
            if first == b"%":
                line = (first + file.readline()).decode(
                    "latin1", errors="replace"
                ).strip()
                header_lines.append(line)
            else:
                event_type = first[0]
                size_byte = file.read(1)
                if not size_byte:
                    raise ValueError("Unable to read the event size.")
                event_size = size_byte[0]
                data_start_position = file.tell()
                break

    for line in header_lines:
        parts = line.split()
        lower = line.lower()
        if lower.startswith("% width"):
            width = int(parts[-1])
        elif lower.startswith("% height"):
            height = int(parts[-1])
        elif lower.startswith("% version") and parts[-1] != "2":
            raise ValueError("This parser currently supports CD DAT Version 2 only.")

    if event_type not in (0, 12):
        raise ValueError(f"Expected CD DAT type 0/12, got {event_type}.")
    if event_size != 8:
        raise ValueError(
            f"This parser supports only 8-byte events; the file uses {event_size} bytes."
        )
    if (file_path.stat().st_size - data_start_position) % event_size:
        raise ValueError("DAT file ends with an incomplete event record.")

    with file_path.open("rb") as file:
        file.seek(data_start_position)
        raw = file.read(event_size)
    if len(raw) != event_size:
        raise ValueError("The DAT file does not contain a complete event.")

    file_first_timestamp = int(np.frombuffer(raw, dtype="<u4")[0])

    return DatHeader(
        event_size=event_size,
        data_start=data_start_position,
        file_first_timestamp=file_first_timestamp,
        width=width,
        height=height,
        event_type=event_type,
        header_lines=tuple(header_lines),
    )


def print_dat_header(header: DatHeader) -> None:
    """Pretty-print header information for diagnostics."""
    print("DAT file header:")
    for line in header.header_lines:
        print(line)
    print(f"Event type: {header.event_type}")
    print(f"Event size: {header.event_size} bytes")
    print(f"Timestamp of the first event: {header.file_first_timestamp} us")
    if header.width is not None and header.height is not None:
        print(f"Sensor resolution: {header.width} x {header.height}")


def decode_raw(raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw 8-byte DAT events into ``timestamp, x, y, polarity`` arrays."""
    usable = (len(raw_bytes) // 8) * 8
    raw_bytes = raw_bytes[:usable]
    if not raw_bytes:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
        )

    words = np.frombuffer(raw_bytes, dtype="<u4")
    timestamps = words[0::2].astype(np.uint64)
    data_words = words[1::2]
    x = (data_words & ((1 << 14) - 1)).astype(np.int32)
    y = ((data_words >> 14) & ((1 << 14) - 1)).astype(np.int32)
    polarity = ((data_words >> 28) & 1).astype(np.uint8)

    if np.any(np.diff(timestamps.astype(np.int64)) < 0):
        raise ValueError(
            "DAT timestamps are not monotonically increasing. "
            "A wrapped/converted timestamp stream is required before analysis."
        )
    return timestamps, x, y, polarity


def find_file_position_at_timestamp(
    file_path: str | Path,
    start_position: int,
    target_timestamp: int,
) -> tuple[int, int, int]:
    """Binary-search the first event whose timestamp is >= ``target_timestamp``.

    Returns ``(byte_position, actual_timestamp, skipped_event_count)``.
    """
    file_path = Path(file_path)
    with file_path.open("rb") as file:
        size = file_path.stat().st_size
        count = (size - int(start_position)) // 8
        if count < 1 or (size - int(start_position)) % 8:
            raise ValueError("No complete DAT events remain after start_position.")

        def at(index: int) -> int:
            file.seek(int(start_position) + index * 8)
            raw = file.read(4)
            if len(raw) != 4:
                raise ValueError("Unexpected end of file during timestamp search.")
            return int(np.frombuffer(raw, dtype="<u4")[0])

        first_time = at(0)
        if target_timestamp <= first_time:
            return int(start_position), first_time, 0

        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi) // 2
            if at(mid) < target_timestamp:
                lo = mid + 1
            else:
                hi = mid

        if lo == count:
            raise ValueError("The target time is later than the end of the DAT recording.")
        return int(start_position) + lo * 8, at(lo), lo


def iter_event_chunks(
    file_path: str | Path,
    start_position: int,
    stop_timestamp: int | None = None,
    chunk_events: int = 1_000_000,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield decoded event chunks without loading the full recording into memory."""
    previous_timestamp = None
    with Path(file_path).open("rb") as file:
        file.seek(start_position)
        while True:
            raw = file.read(int(chunk_events) * 8)
            timestamps, x, y, polarity = decode_raw(raw)
            if len(timestamps) == 0:
                return

            if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
                raise ValueError("DAT timestamps decrease across chunks.")
            previous_timestamp = int(timestamps[-1])

            if stop_timestamp is None:
                yield timestamps, x, y, polarity
                continue

            keep_count = int(np.searchsorted(timestamps, stop_timestamp, side="left"))
            if keep_count > 0:
                yield (
                    timestamps[:keep_count],
                    x[:keep_count],
                    y[:keep_count],
                    polarity[:keep_count],
                )
            if keep_count < len(timestamps):
                return


def scan_polarity_bins(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    duration_us: int,
    bin_us: int,
    max_bins: int = 2_000_000,
    chunk_events: int = 1_000_000,
) -> tuple[np.ndarray, int | None]:
    """Count OFF/ON events in fixed-width time bins.

    The returned array has shape ``(n_bins, 2)`` with columns ``OFF, ON``.
    """
    import math

    n = int(math.ceil(duration_us / bin_us))
    if n <= 0 or n > max_bins:
        raise ValueError("The requested timing histogram is too large or invalid.")

    counts = np.zeros((n, 2), np.int64)
    last_timestamp = None
    stop_timestamp = int(actual_start_timestamp) + int(duration_us)

    for timestamps, _, _, polarity in iter_event_chunks(
        file_path,
        read_start_position,
        stop_timestamp=stop_timestamp,
        chunk_events=chunk_events,
    ):
        if not len(timestamps):
            continue
        last_timestamp = int(timestamps[-1])
        bins = (
            timestamps.astype(np.int64) - int(actual_start_timestamp)
        ) // int(bin_us)
        valid = (bins >= 0) & (bins < n)
        flat = bins[valid] * 2 + polarity[valid]
        counts += np.bincount(flat, minlength=n * 2).reshape(n, 2)

    return counts, last_timestamp


class DisplayEventReader:
    """Sliding event-count reader for interactive reference frames and videos.

    The reader updates a pair of ON/OFF count images when adjacent display
    windows overlap, avoiding full rescans of the DAT file.
    """

    def __init__(
        self,
        file_path: str | Path,
        start_position: int,
        width: int,
        height: int,
        chunk_events: int = 1_000_000,
    ) -> None:
        self.file_path = Path(file_path)
        self.start_position = int(start_position)
        self.width = int(width)
        self.height = int(height)
        self.chunk_events = int(chunk_events)

        size = self.file_path.stat().st_size - self.start_position
        if size <= 0 or size % 8:
            raise ValueError("Display reader start does not point to complete DAT events.")

        self.event_count = size // 8
        self.file = self.file_path.open("rb")
        self.first_timestamp = self._timestamp(0)
        self.last_timestamp = self._timestamp(self.event_count - 1)
        self.counts = np.zeros((2, self.height, self.width), dtype=np.int64)
        self.range_start = 0
        self.range_end = 0

    def __enter__(self) -> "DisplayEventReader":
        return self

    def __exit__(self, *args) -> None:
        self.file.close()

    def _timestamp(self, index: int) -> int:
        self.file.seek(self.start_position + int(index) * 8)
        raw = self.file.read(4)
        if len(raw) != 4:
            raise ValueError("Unexpected end of DAT file.")
        return int.from_bytes(raw, "little")

    def lower_bound(self, timestamp: int) -> int:
        if timestamp <= self.first_timestamp:
            return 0
        if timestamp > self.last_timestamp:
            return self.event_count

        lo, hi = 0, self.event_count
        while lo < hi:
            mid = (lo + hi) // 2
            if self._timestamp(mid) < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _accumulate(self, first: int, stop: int, sign: int) -> None:
        self.file.seek(self.start_position + first * 8)
        remaining = stop - first
        while remaining > 0:
            take = min(self.chunk_events, remaining)
            raw = self.file.read(take * 8)
            if len(raw) != take * 8:
                raise ValueError("DAT length changed or contains an incomplete event.")
            _, x, y, p = decode_raw(raw)
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            np.add.at(self.counts, (p[valid], y[valid], x[valid]), sign)
            remaining -= take

    def window_counts(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ON image, OFF image and number of raw events in ``[start, end)``."""
        if end_timestamp < start_timestamp:
            raise ValueError("Display window end must not precede its start.")

        first = self.lower_bound(start_timestamp)
        stop = self.lower_bound(end_timestamp)

        if self.range_start <= first <= self.range_end <= stop:
            self._accumulate(self.range_start, first, -1)
            self._accumulate(self.range_end, stop, +1)
        elif (first, stop) != (self.range_start, self.range_end):
            self.counts.fill(0)
            self._accumulate(first, stop, +1)

        self.range_start = first
        self.range_end = stop
        return self.counts[1], self.counts[0], stop - first
