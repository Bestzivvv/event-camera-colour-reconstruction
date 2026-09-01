"""Automatic first-flash detection and timing diagnostics.

This module scans the retained DAT recording, keeps long silent intervals, and
searches for one candidate R1 onset whose *entire fixed RGB schedule* is
supported by the event statistics.  It does not independently move later
flashes to local maxima.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from dat_decoder import (
    find_file_position_at_timestamp,
    iter_event_chunks,
    scan_polarity_bins,
)
from temporal_alignment import (
    TimingConfig,
    TimingDetectionError,
    build_fixed_flash_onsets,
    detect_first_flash_automatically,
)


@dataclass(frozen=True)
class FlashSearchConfig:
    first_red_search_duration_us: int | None = None
    detection_block_duration_us: int = 1_000_000
    overview_bin_us: int = 1_000
    max_detection_bins: int = 2_000_000
    chunk_events: int = 1_000_000


@dataclass
class FlashDetectionResult:
    onsets_us: np.ndarray
    centres_us: np.ndarray
    total_counts: np.ndarray
    smoothed_on_counts: np.ndarray
    polarity_counts: np.ndarray
    details: dict
    recording_overview: dict


def _save_initial_activity(
    output_dir: Path,
    centres_us: np.ndarray,
    counts: np.ndarray,
    actual_read_start_us: int,
    bin_us: int,
) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, constrained_layout=True
    )
    axes[0].plot(centres_us, counts[:, 1], label="Raw ON", lw=0.65)
    axes[0].plot(centres_us, counts[:, 0], label="Raw OFF", lw=0.65, alpha=0.7)
    axes[0].set(
        ylabel=f"Events / {bin_us} us",
        title="Full recording overview (not an LED-brightness trace)",
    )

    block = max(1, int(math.ceil(10_000 / bin_us)))
    starts = np.arange(0, len(counts), block)
    summed = np.add.reduceat(counts, starts, axis=0)
    widths = np.minimum(block, len(counts) - starts)
    x = (starts + widths / 2) * bin_us
    axes[1].plot(x, summed[:, 1], label="Binned ON", lw=0.8)
    axes[1].plot(x, summed[:, 0], label="Binned OFF", lw=0.8, alpha=0.7)
    axes[1].set(
        ylabel=f"Events / up to {block * bin_us} us",
        xlabel="Time since read start (us)",
        title=f"Read start = {actual_read_start_us} us after first DAT event",
    )
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.grid(alpha=0.15)
        axis.legend()

    figure.savefig(output_dir / "initial_activity.png", dpi=160)
    plt.close(figure)

    with (output_dir / "initial_activity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "bin_start_after_read_us",
                "bin_start_after_file_first_us",
                "ON",
                "OFF",
                "TOTAL",
                "bin_us",
            ]
        )
        for i, row in enumerate(counts):
            t = int(i * bin_us)
            writer.writerow(
                [
                    t,
                    int(actual_read_start_us + t),
                    int(row[1]),
                    int(row[0]),
                    int(row.sum()),
                    bin_us,
                ]
            )


def scan_full_recording_overview(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    actual_read_start_us: int,
    output_dir: str | Path,
    search_config: FlashSearchConfig,
) -> tuple[int, dict]:
    """Validate chronological order and create a coarse whole-file overview."""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    with file_path.open("rb") as file:
        file.seek(-8, 2)
        last_raw = int(np.frombuffer(file.read(4), dtype="<u4")[0])

    duration_us = last_raw - int(actual_start_timestamp) + 1
    if duration_us <= 0:
        raise ValueError("Recording end precedes the selected read start.")

    coarse_bin_us = max(
        int(search_config.overview_bin_us),
        int(math.ceil(duration_us / 200_000)),
    )
    counts = np.zeros(
        (int(math.ceil(duration_us / coarse_bin_us)), 2), dtype=np.int64
    )

    total = 0
    previous = None
    last_update = time.monotonic()
    max_gap_us = 0
    gap_pair = [0, 0]

    for timestamps, _, _, polarity in iter_event_chunks(
        file_path,
        read_start_position,
        chunk_events=search_config.chunk_events,
    ):
        t = timestamps.astype(np.int64) - int(actual_start_timestamp)
        if np.any(t < 0) or np.any(t >= duration_us):
            raise ValueError("DAT timestamps fall outside the expected first/last range.")

        delta = np.diff(t, prepend=int(t[0]) if previous is None else previous)
        largest = int(np.argmax(delta))
        if int(delta[largest]) > max_gap_us:
            max_gap_us = int(delta[largest])
            gap_pair = [int(t[largest] - max_gap_us), int(t[largest])]
        previous = int(t[-1])

        bins = t // coarse_bin_us
        lo, hi = int(bins[0]), int(bins[-1]) + 1
        counts[lo:hi] += np.bincount(
            (bins - lo) * 2 + polarity,
            minlength=(hi - lo) * 2,
        ).reshape(-1, 2)
        total += len(t)

        if time.monotonic() - last_update >= 3:
            print(
                f"Full-recording overview: {total:,} events, "
                f"{t[-1] / 1e6:.3f}/{duration_us / 1e6:.3f}s",
                flush=True,
            )
            last_update = time.monotonic()

    centres = (np.arange(len(counts)) + 0.5) * coarse_bin_us
    _save_initial_activity(
        output_dir,
        centres,
        counts,
        actual_read_start_us,
        coarse_bin_us,
    )

    overview = {
        "retained_event_span_us": duration_us - 1,
        "available_end_after_read_us": duration_us,
        "events": int(total),
        "ON": int(counts[:, 1].sum()),
        "OFF": int(counts[:, 0].sum()),
        "overview_bin_us": coarse_bin_us,
        "max_interevent_gap_us": max_gap_us,
        "gap_between_events_after_read_us": gap_pair,
        "note": (
            "Whole-file coarse overview. Quiet bins are retained and are not "
            "used as a local peak locator."
        ),
    }
    (output_dir / "recording_summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return duration_us, overview


def _local_details_to_read_time(details: dict, origin_us: int) -> dict:
    details = dict(details)
    for key in (
        "rejected_best_candidate_after_read_us",
        "first_onset_before_sync_us",
        "first_onset_after_sync_us",
    ):
        if key in details:
            details[key] = int(details[key]) + int(origin_us)
    if "near_best_start_range_us" in details:
        details["near_best_start_range_us"] = [
            int(t) + int(origin_us) for t in details["near_best_start_range_us"]
        ]
    details["fine_block_origin_after_read_us"] = int(origin_us)
    return details


def detect_flash_onsets_chunked(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    actual_read_start_us: int,
    output_dir: str | Path,
    timing_config: TimingConfig,
    search_config: FlashSearchConfig = FlashSearchConfig(),
) -> FlashDetectionResult:
    """Search the complete recording for an automatically supported R1 onset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_end_us, overview = scan_full_recording_overview(
        file_path,
        read_start_position,
        actual_start_timestamp,
        actual_read_start_us,
        output_dir,
        search_config,
    )

    last_offset = int(build_fixed_flash_onsets(0, timing_config)[-1])
    horizon_us = last_offset + int(timing_config.flash_period_us)
    quiet_us = max(
        int(timing_config.first_red_baseline_us),
        3 * int(timing_config.flash_period_us)
        + int(timing_config.cycle_extra_delay_us),
    )
    bin_us = int(timing_config.flash_detection_bin_us)
    first_max_us = available_end_us - horizon_us
    if search_config.first_red_search_duration_us is not None:
        first_max_us = min(
            first_max_us, int(search_config.first_red_search_duration_us)
        )

    pad_us = 2 * int(timing_config.flash_period_us) + abs(
        int(timing_config.sync_offset_us)
    )
    capacity_us = (
        int(search_config.max_detection_bins) * bin_us
        - quiet_us
        - horizon_us
        - 2 * pad_us
        - 2 * bin_us
    )
    stride_us = (
        min(int(search_config.detection_block_duration_us), capacity_us)
        // bin_us
        * bin_us
    )
    if stride_us < bin_us:
        raise ValueError("max_detection_bins is too small for the fixed RGB schedule.")

    print(
        f"Fine timing search: 0..{max(0, first_max_us) / 1e6:.6f}s, "
        f"bin={bin_us} us. Long silent gaps are retained.",
        flush=True,
    )

    report_path = output_dir / "automatic_timing_report.json"
    best_error: dict = {}
    best_trace = None
    selected = None
    blocks_checked = 0

    with (output_dir / "fine_search_blocks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as log:
        writer = csv.writer(log)
        writer.writerow(
            [
                "core_start_after_read_us",
                "core_end_after_read_us",
                "events_in_context",
                "status",
            ]
        )

        for core_start in range(0, max(0, first_max_us) + 1, stride_us):
            blocks_checked += 1
            core_end = min(first_max_us, core_start + stride_us)
            left = max(0, core_start - quiet_us - pad_us) // bin_us * bin_us
            right = min(
                available_end_us,
                core_start + stride_us + pad_us + horizon_us,
            )
            if right - left < quiet_us + horizon_us:
                writer.writerow([core_start, core_end, 0, "insufficient_context"])
                continue

            position, _, _ = find_file_position_at_timestamp(
                file_path,
                read_start_position,
                int(actual_start_timestamp) + left,
            )
            counts, _ = scan_polarity_bins(
                file_path,
                position,
                int(actual_start_timestamp) + left,
                right - left,
                bin_us,
                max_bins=search_config.max_detection_bins,
                chunk_events=search_config.chunk_events,
            )
            centres = left + (np.arange(len(counts)) + 0.5) * bin_us
            number = int(counts.sum())

            if not number:
                writer.writerow([core_start, core_end, 0, "quiet_continue"])
                continue

            try:
                local_first, details, checks = detect_first_flash_automatically(
                    counts,
                    timing_config,
                    available_end_us=right - left,
                    candidate_min_us=max(0, core_start - pad_us - left),
                    candidate_max_us=(
                        min(first_max_us, core_start + stride_us + pad_us) - left
                    ),
                )
            except TimingDetectionError as error:
                details = _local_details_to_read_time(error.diagnostics, left)
                writer.writerow(
                    [
                        core_start,
                        core_end,
                        number,
                        "no_supported_first_flash_continue",
                    ]
                )
                if details.get("rank_score", -math.inf) > best_error.get(
                    "rank_score", -math.inf
                ):
                    best_error = dict(details)
                    best_error["local_rejection"] = str(error)
                    candidate = int(
                        details.get(
                            "rejected_best_candidate_after_read_us",
                            details.get("first_onset_before_sync_us", left),
                        )
                    )
                    mask = (
                        (centres >= candidate - max(2000, 2 * timing_config.flash_period_us))
                        & (
                            centres
                            <= candidate + horizon_us + timing_config.flash_period_us
                        )
                    )
                    best_trace = (centres[mask].copy(), counts[mask].copy())
                continue

            first_onset = int(local_first + left)
            details = _local_details_to_read_time(details, left)
            for item in checks:
                item["onset_after_read_us"] += int(left)
            selected = (first_onset, details, checks, centres, counts)
            writer.writerow(
                [core_start, core_end, number, "statistical_timing_check_passed"]
            )
            break

    common = {
        "flash_period_us": int(timing_config.flash_period_us),
        "fade_duration_us": int(timing_config.fade_duration_us),
        "dark_interval_us": int(timing_config.dark_interval_us),
        "recording": overview,
        "first_red_search_duration_us": search_config.first_red_search_duration_us,
        "search_last_eligible_onset_us": int(first_max_us),
        "fine_blocks_checked": blocks_checked,
        "timing_basis": (
            "Configured fixed timing; no independent per-flash peak shifts."
        ),
    }

    if selected is None:
        details = dict(best_error, **common)
        details.update(
            {
                "automatic_timing_passed": False,
                "error": (
                    "No reliable first flash was found across the configured search range."
                ),
            }
        )
        report_path.write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if best_trace is not None:
            centres, trace = best_trace
            figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
            axis.plot(centres / 1000, trace[:, 1], lw=0.7, label="ON")
            axis.plot(centres / 1000, trace[:, 0], lw=0.7, label="OFF")
            axis.set(
                xlabel="Time after read start (ms)",
                ylabel=f"Events / {bin_us} us",
                title="Rejected activity candidate: no verified fixed RGB schedule",
            )
            axis.legend()
            figure.savefig(output_dir / "rejected_timing.png", dpi=160)
            plt.close(figure)

        raise TimingDetectionError(details["error"], details)

    first_onset, details, checks, centres, counts = selected
    onsets = build_fixed_flash_onsets(first_onset, timing_config)
    final_response_end = int(
        onsets[-1]
        + timing_config.response_delay_us
        + timing_config.response_window_us
    )
    if final_response_end > available_end_us:
        raise TimingDetectionError("Recording does not cover every RGB response window.")

    details.update(common)
    details.update(
        {
            "source": "full_recording_automatic_first_activity_fixed_RGB_check",
            "sync_offset_us": int(timing_config.sync_offset_us),
            "cycle_extra_delay_us": int(timing_config.cycle_extra_delay_us),
            "response_delay_us": int(timing_config.response_delay_us),
            "response_window_us": int(timing_config.response_window_us),
        }
    )
    report_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if checks:
        with (output_dir / "automatic_timing_check.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)

    smooth = np.convolve(counts[:, 1].astype(float), [0.2, 0.6, 0.2], mode="same")
    return FlashDetectionResult(
        onsets_us=onsets,
        centres_us=centres,
        total_counts=counts.sum(axis=1),
        smoothed_on_counts=smooth,
        polarity_counts=counts,
        details=details,
        recording_overview=overview,
    )


def save_flash_figure(
    result: FlashDetectionResult,
    actual_read_start_us: int,
    output_dir: str | Path,
    timing_config: TimingConfig,
    show: bool = False,
) -> None:
    """Save ``flash_detection.png`` and detailed timing diagnostics."""
    output_dir = Path(output_dir)
    onsets = result.onsets_us
    centres = result.centres_us
    trace = result.polarity_counts
    colours = {"R": "#cb4040", "G": "#299557", "B": "#3d72d5"}
    onset0 = int(onsets[0])

    def draw(axis, left, right, title):
        selected = (centres >= left) & (centres <= right)
        x = (centres[selected] - onset0) / 1000
        axis.plot(x, trace[selected, 1], color="#333333", lw=0.85, label="Raw ON")
        axis.plot(x, trace[selected, 0], color="#c77b27", lw=0.65, alpha=0.75, label="Raw OFF")

        for i, (colour, onset) in enumerate(zip(timing_config.flash_colours, onsets)):
            if onset < left or onset >= right:
                continue
            x0 = (onset - onset0) / 1000
            axis.axvline(x0, color=colours[colour], ls="--", lw=0.9)
            axis.axvspan(
                (onset + timing_config.response_delay_us - onset0) / 1000,
                (
                    onset
                    + timing_config.response_delay_us
                    + timing_config.response_window_us
                    - onset0
                ) / 1000,
                color=colours[colour],
                alpha=0.14,
            )
            axis.axvline(
                (onset + timing_config.fade_duration_us - onset0) / 1000,
                color=colours[colour],
                ls=":",
                lw=0.85,
            )
            axis.text(
                x0,
                0.98,
                f"{colour}{i // 3 + 1}",
                transform=axis.get_xaxis_transform(),
                va="top",
                color=colours[colour],
                fontsize=9,
                clip_on=True,
            )

        axis.set(
            xlim=((left - onset0) / 1000, (right - onset0) / 1000),
            title=title,
            xlabel="Time relative to candidate R1 (ms)",
            ylabel=f"Events / {timing_config.flash_detection_bin_us} us",
        )
        axis.grid(alpha=0.15)
        axis.set_ylim(bottom=0)
        axis.legend(loc="upper right", fontsize=8)

    left = max(0, onset0 - 2 * timing_config.flash_period_us)
    right = min(centres[-1], onsets[-1] + timing_config.flash_period_us)
    figure, axis = plt.subplots(figsize=(13, 4), constrained_layout=True)
    draw(
        axis,
        left,
        right,
        (
            f"Automatic timing check passed: fade={timing_config.fade_duration_us} us, "
            f"period={timing_config.flash_period_us} us"
        ),
    )
    figure.savefig(output_dir / "flash_detection.png", dpi=170)

    with (output_dir / "flash_timing.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "stage",
                "predicted_onset_after_read_us",
                "predicted_onset_after_file_first_us",
                "response_start_after_read_us",
                "response_end_after_read_us",
            ]
        )
        for i, (colour, onset) in enumerate(
            zip(timing_config.flash_colours, onsets)
        ):
            response_start = int(onset + timing_config.response_delay_us)
            writer.writerow(
                [
                    f"{colour}{i // 3 + 1}",
                    int(onset),
                    int(actual_read_start_us + onset),
                    response_start,
                    response_start + int(timing_config.response_window_us),
                ]
            )

    if show:
        plt.show()
    plt.close(figure)
