"""Quantitative illumination-duration experiment for the event-camera project.

Purpose
-------
Compare the same scene and the same ROI polygons across four illumination
durations:

    r1: 100 ms illumination + 500 us dark
    r2:  10 ms illumination + 500 us dark
    r3:   1 ms illumination + 500 us dark
    r4: 500 us illumination + 500 us dark

The script performs two analyses for every recording:

1. FULL window
   The quantitative response window equals the illumination/fade duration.

2. FIXED_500US window
   Every recording is measured using only the first 500 us after each
   predicted RGB onset.  This separates early transient behaviour from
   long-window temporal/PWM accumulation.

For each analysis it exports:
    - raw ON/OFF measurements,
    - time-normalised event rates,
    - ON_PEAK response vectors (baseline method),
    - ON_MEAN response vectors,
    - ON_MEDIAN response vectors,
    - positive NET_MEAN response vectors,
    - positive NET_MEDIAN response vectors,
    - repeatability statistics,
    - ROI separability,
    - cross-duration response distances,
    - comparison plots.

Important
---------
The same polygon coordinates are reused for all recordings.  This is only a
controlled comparison if the camera, object and scene geometry remained fixed
between r1-r4.  Inspect the automatically saved ROI alignment-check images.

Run from the repository root:

    python run_duration_experiment.py

This script uses the existing modules in src/ and does not replace the normal
run_pipeline.py workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
import csv
import json
import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Repository imports
# =============================================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dat_decoder import (  # noqa: E402
    find_file_position_at_timestamp,
    print_dat_header,
    read_dat_header,
)
from temporal_alignment import TimingConfig  # noqa: E402
from flash_detection import (  # noqa: E402
    FlashSearchConfig,
    detect_flash_onsets_chunked,
    save_flash_figure,
)
from roi_analysis import (  # noqa: E402
    ROIConfig,
    choose_polygon_rois,
    configure_rois,
    count_roi_flash_responses_chunked,
    make_roi_masks,
    prepare_analysis_roi_masks,
    save_effective_roi_mask_figure,
    save_png,
    select_clear_reference_frame,
)


# =============================================================================
# User settings
# =============================================================================

@dataclass(frozen=True)
class Experiment:
    name: str
    data_file: Path
    illumination_us: int
    dark_us: int = 500
    measured_period_us: int | None = None

    @property
    def period_us(self) -> int:
        if self.measured_period_us is not None:
            return int(self.measured_period_us)
        return int(self.illumination_us + self.dark_us)


# -------------------------------------------------------------------------
# EDIT ONLY THESE FOUR PATHS.
# The duration mapping below already matches your r1-r4 experiment.
# -------------------------------------------------------------------------

EXPERIMENTS = (
    Experiment(
        "r1_100ms",
        Path(r"D:\RGB Events\DATASET\new\orange_new\r_1.dat"),
        illumination_us=100_000,
    ),
    Experiment(
        "r2_10ms",
        Path(r"D:\RGB Events\DATASET\new\orange_new\r_2.dat"),
        illumination_us=10_000,
    ),
    Experiment(
        "r3_1ms",
        Path(r"D:\RGB Events\DATASET\new\orange_new\r_3.dat"),
        illumination_us=1_000,
    ),
    Experiment(
        "r4_500us",
        Path(r"D:\RGB Events\DATASET\new\orange_new\r_4.dat"),
        illumination_us=500,
    ),
)

# Use r3 to draw the common ROI polygons once.
ROI_REFERENCE_EXPERIMENT = "r3_1ms"

# Set to an integer (e.g. 3) to avoid being asked at run time.
# None = ask interactively.
NUM_ROIS: int | None = 3

# The second experiment analysis always uses the first 500 us after onset.
FIXED_RESPONSE_WINDOW_US = 500

NUM_RGB_REPEATS = 3
SKIP_BEFORE_TIME_US = 0

# Do not use display-only material brightness in this quantitative experiment.
# This script does not generate pseudo-colour GIFs.

OUTPUT_ROOT = ROOT / "duration_experiment_results"

# If this JSON already exists, the same polygons are loaded without redrawing.
# Delete it if the object/camera geometry has changed and you need new ROIs.
ROI_POLYGON_FILE = ROOT / "duration_experiment_rois.json"

# Plot every ROI.  If False, plots only ROI_1.
PLOT_ALL_ROIS = True


# =============================================================================
# Shared configurations
# =============================================================================

FLASH_SEARCH = FlashSearchConfig(
    first_red_search_duration_us=None,
    detection_block_duration_us=1_000_000,
    overview_bin_us=1_000,
    max_detection_bins=2_000_000,
    chunk_events=1_000_000,
)

ROI_CONFIG_INTERACTIVE = ROIConfig(
    num_rois=NUM_ROIS,
    nested_roi_mode="EXCLUDE_CONTAINED",
    reference_window_us=50_000,
    reference_step_us=10_000,
    reference_search_duration_us=300_000,
    reference_start_after_read_us=None,
    reference_clip_value=1,
    reference_display_mode="TOTAL",
    display_scale=1.5,
    event_point_size=1,
    automatic_reference_frame=False,
    chunk_events=1_000_000,
)

ROI_CONFIG_AUTO = ROIConfig(
    num_rois=NUM_ROIS,
    nested_roi_mode="EXCLUDE_CONTAINED",
    reference_window_us=50_000,
    reference_step_us=10_000,
    reference_search_duration_us=300_000,
    reference_start_after_read_us=None,
    reference_clip_value=1,
    reference_display_mode="TOTAL",
    display_scale=1.5,
    event_point_size=1,
    automatic_reference_frame=True,
    chunk_events=1_000_000,
)


# =============================================================================
# Utility functions
# =============================================================================

COLOURS = ("R", "G", "B")
ESTIMATORS = (
    "ON_PEAK",
    "ON_MEAN",
    "ON_MEDIAN",
    "NET_POS_MEAN",
    "NET_POS_MEDIAN",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalise_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    return values / total if total > 0 else np.zeros_like(values)


def make_timing_config(
    experiment: Experiment,
    response_window_us: int,
) -> TimingConfig:
    """Construct a timing configuration for one recording."""
    fade_us = int(experiment.illumination_us)
    dark_us = int(experiment.dark_us)
    period_us = int(experiment.period_us)

    detection_bin_us = max(1, min(10, fade_us // 20))
    edge_guard_us = max(
        detection_bin_us,
        min(50, dark_us // 5, max(1, fade_us // 5)),
    )

    if response_window_us > period_us:
        raise ValueError(
            f"{experiment.name}: response window {response_window_us} us "
            f"is longer than stage period {period_us} us."
        )

    return TimingConfig(
        num_rgb_repeats=NUM_RGB_REPEATS,
        fade_duration_us=fade_us,
        peak_apply_time_us=0,
        dark_interval_us=dark_us,
        flash_period_us=period_us,
        cycle_extra_delay_us=0,
        sync_offset_us=0,
        response_delay_us=0,
        response_window_us=int(response_window_us),
        flash_detection_bin_us=detection_bin_us,
        first_red_baseline_us=1_000,
        timing_min_flash_score=2.5,
        timing_min_sequence_score=5.0,
        timing_min_supported_fraction=2 / 3,
        timing_max_startup_score=3.0,
        timing_edge_guard_us=edge_guard_us,
    )


def open_recording(experiment: Experiment) -> dict:
    """Read the DAT header and resolve the configured read start."""
    if not experiment.data_file.is_file():
        raise FileNotFoundError(
            f"{experiment.name}: DAT file not found:\n{experiment.data_file}\n"
            "Edit the EXPERIMENTS paths in run_duration_experiment.py."
        )

    header = read_dat_header(experiment.data_file)
    print(f"\n=== {experiment.name} ===")
    print_dat_header(header)

    if header.width is None or header.height is None:
        raise ValueError(f"{experiment.name}: DAT header is missing Width/Height.")

    target = header.file_first_timestamp + int(SKIP_BEFORE_TIME_US)
    read_start_position, actual_start_timestamp, skipped_count = (
        find_file_position_at_timestamp(
            experiment.data_file,
            header.data_start,
            target,
        )
    )
    actual_read_start_us = actual_start_timestamp - header.file_first_timestamp

    return {
        "header": header,
        "read_start_position": int(read_start_position),
        "actual_start_timestamp": int(actual_start_timestamp),
        "actual_read_start_us": int(actual_read_start_us),
        "skipped_count": int(skipped_count),
        "width": int(header.width),
        "height": int(header.height),
    }


def detect_timing(
    experiment: Experiment,
    recording: dict,
    output_dir: Path,
) -> tuple[object, TimingConfig]:
    """Detect one initial phase using the FULL-window timing configuration."""
    timing = make_timing_config(
        experiment,
        response_window_us=experiment.illumination_us,
    )

    detection = detect_flash_onsets_chunked(
        experiment.data_file,
        recording["read_start_position"],
        recording["actual_start_timestamp"],
        recording["actual_read_start_us"],
        output_dir,
        timing,
        FLASH_SEARCH,
    )

    save_flash_figure(
        detection,
        recording["actual_read_start_us"],
        output_dir,
        timing,
        show=False,
    )

    return detection, timing


def save_polygons(path: Path, polygons: dict[str, np.ndarray], width: int, height: int) -> None:
    payload = {
        "width": int(width),
        "height": int(height),
        "polygons": {
            name: np.asarray(points, dtype=int).tolist()
            for name, points in polygons.items()
        },
        "note": (
            "Same native-sensor polygon coordinates are reused across r1-r4. "
            "Only valid if scene geometry is unchanged."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_polygons(path: Path, width: int, height: int) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["width"]) != width or int(payload["height"]) != height:
        raise ValueError(
            "Saved ROI polygon resolution does not match the current recordings."
        )
    polygons = {
        name: np.asarray(points, dtype=np.int32)
        for name, points in payload["polygons"].items()
    }
    return polygons


def select_or_load_common_rois(
    reference_experiment: Experiment,
    recording: dict,
    detection,
    output_dir: Path,
) -> tuple[
    list[tuple[str, str, tuple[int, int, int]]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, int],
]:
    """Draw the common ROIs once, or load previously saved polygons."""
    width = recording["width"]
    height = recording["height"]

    if ROI_POLYGON_FILE.is_file():
        polygons = load_polygons(ROI_POLYGON_FILE, width, height)
        roi_definitions = configure_rois(len(polygons))
        expected_names = [item[0] for item in roi_definitions]
        if set(polygons) != set(expected_names):
            raise ValueError(
                f"Saved ROI names {sorted(polygons)} do not match expected "
                f"{expected_names}. Delete {ROI_POLYGON_FILE} and redraw."
            )

        # Still create an automatic reference image for visual verification.
        reference_gray, _, _ = select_clear_reference_frame(
            reference_experiment.data_file,
            recording["read_start_position"],
            recording["actual_start_timestamp"],
            recording["actual_read_start_us"],
            width,
            height,
            first_onset_us=int(detection.onsets_us[0]),
            config=ROI_CONFIG_AUTO,
        )
    else:
        roi_definitions = configure_rois(ROI_CONFIG_INTERACTIVE.num_rois)
        reference_gray, _, _ = select_clear_reference_frame(
            reference_experiment.data_file,
            recording["read_start_position"],
            recording["actual_start_timestamp"],
            recording["actual_read_start_us"],
            width,
            height,
            first_onset_us=int(detection.onsets_us[0]),
            config=ROI_CONFIG_INTERACTIVE,
        )
        polygons = choose_polygon_rois(
            reference_gray,
            roi_definitions,
            display_scale=ROI_CONFIG_INTERACTIVE.display_scale,
        )
        save_polygons(ROI_POLYGON_FILE, polygons, width, height)
        print(f"Saved common ROI polygons: {ROI_POLYGON_FILE}")

    save_png(output_dir / "common_roi_reference.png", reference_gray)

    raw_masks, _ = make_roi_masks(polygons, width, height)
    effective_masks, effective_areas, _ = prepare_analysis_roi_masks(
        raw_masks,
        roi_definitions,
        nested_roi_mode=ROI_CONFIG_INTERACTIVE.nested_roi_mode,
    )
    save_effective_roi_mask_figure(
        reference_gray,
        polygons,
        effective_masks,
        output_dir / "common_roi_effective_masks.png",
    )

    return roi_definitions, polygons, effective_masks, effective_areas


def make_alignment_check(
    experiment: Experiment,
    recording: dict,
    detection,
    polygons: dict[str, np.ndarray],
    effective_masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save an automatic reference view with the same ROIs for alignment checking."""
    gray, _, _ = select_clear_reference_frame(
        experiment.data_file,
        recording["read_start_position"],
        recording["actual_start_timestamp"],
        recording["actual_read_start_us"],
        recording["width"],
        recording["height"],
        first_onset_us=int(detection.onsets_us[0]),
        config=ROI_CONFIG_AUTO,
    )
    save_effective_roi_mask_figure(
        gray,
        polygons,
        effective_masks,
        output_path,
    )


def add_derived_measurements(
    rows: list[dict],
    experiment: Experiment,
    window_mode: str,
) -> list[dict]:
    """Add time-normalised rates and experiment labels to raw response rows."""
    result = []
    for row in rows:
        item = dict(row)
        window_us = float(item["response_window_us"])
        window_ms = window_us / 1000.0

        on_density = float(item["on_density"])
        off_density = float(item["off_density"])
        net_density = on_density - off_density

        item.update(
            {
                "experiment": experiment.name,
                "illumination_us": int(experiment.illumination_us),
                "illumination_ms": experiment.illumination_us / 1000.0,
                "dark_us": int(experiment.dark_us),
                "stage_period_us": int(experiment.period_us),
                "window_mode": window_mode,
                "net_density": net_density,
                "net_positive_density": max(net_density, 0.0),
                "on_rate_per_pixel_ms": on_density / window_ms,
                "off_rate_per_pixel_ms": off_density / window_ms,
                "net_rate_per_pixel_ms": net_density / window_ms,
                "net_positive_rate_per_pixel_ms": max(net_density, 0.0) / window_ms,
            }
        )
        result.append(item)
    return result


def aggregate_values(values: np.ndarray, estimator: str) -> float:
    if estimator.endswith("PEAK"):
        return float(np.max(values))
    if estimator.endswith("MEAN"):
        return float(np.mean(values))
    if estimator.endswith("MEDIAN"):
        return float(np.median(values))
    raise ValueError(f"Unknown estimator: {estimator}")


def calculate_response_summaries(
    rows: list[dict],
    estimator: str,
) -> list[dict]:
    """Calculate one RGB vector per ROI for a selected estimator."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unsupported estimator: {estimator}")

    use_net = estimator.startswith("NET_POS_")
    density_key = "net_positive_density" if use_net else "on_density"
    rate_key = "net_positive_rate_per_pixel_ms" if use_net else "on_rate_per_pixel_ms"

    experiment_name = rows[0]["experiment"]
    illumination_us = int(rows[0]["illumination_us"])
    window_mode = rows[0]["window_mode"]
    response_window_us = int(rows[0]["response_window_us"])

    summaries = []
    for roi_name in sorted({row["roi"] for row in rows}):
        amplitude_density = []
        amplitude_rate = []
        repeat_stds = []

        for colour in COLOURS:
            colour_rows = [
                row for row in rows
                if row["roi"] == roi_name and row["colour"] == colour
            ]
            density_values = np.asarray(
                [float(row[density_key]) for row in colour_rows],
                dtype=np.float64,
            )
            rate_values = np.asarray(
                [float(row[rate_key]) for row in colour_rows],
                dtype=np.float64,
            )

            amplitude_density.append(aggregate_values(density_values, estimator))
            amplitude_rate.append(aggregate_values(rate_values, estimator))
            repeat_stds.append(
                float(np.std(density_values, ddof=1))
                if len(density_values) > 1
                else 0.0
            )

        amplitude_density = np.asarray(amplitude_density, dtype=np.float64)
        amplitude_rate = np.asarray(amplitude_rate, dtype=np.float64)

        # The response composition is based on channel amplitudes.
        vector = normalise_nonnegative(amplitude_density)

        summaries.append(
            {
                "experiment": experiment_name,
                "illumination_us": illumination_us,
                "illumination_ms": illumination_us / 1000.0,
                "window_mode": window_mode,
                "response_window_us": response_window_us,
                "roi": roi_name,
                "estimator": estimator,
                "amplitude_R_density": float(amplitude_density[0]),
                "amplitude_G_density": float(amplitude_density[1]),
                "amplitude_B_density": float(amplitude_density[2]),
                "amplitude_R_rate_per_pixel_ms": float(amplitude_rate[0]),
                "amplitude_G_rate_per_pixel_ms": float(amplitude_rate[1]),
                "amplitude_B_rate_per_pixel_ms": float(amplitude_rate[2]),
                "vector_R": float(vector[0]),
                "vector_G": float(vector[1]),
                "vector_B": float(vector[2]),
                "repeat_std_R": repeat_stds[0],
                "repeat_std_G": repeat_stds[1],
                "repeat_std_B": repeat_stds[2],
            }
        )

    return summaries


def calculate_repeatability(rows: list[dict]) -> list[dict]:
    """Calculate ON and signed-NET repeatability for every channel."""
    output = []
    experiment_name = rows[0]["experiment"]
    illumination_us = int(rows[0]["illumination_us"])
    window_mode = rows[0]["window_mode"]
    response_window_us = int(rows[0]["response_window_us"])

    for roi_name in sorted({row["roi"] for row in rows}):
        for colour in COLOURS:
            selected = [
                row for row in rows
                if row["roi"] == roi_name and row["colour"] == colour
            ]

            on = np.asarray([row["on_density"] for row in selected], dtype=float)
            net = np.asarray([row["net_density"] for row in selected], dtype=float)
            on_rate = np.asarray(
                [row["on_rate_per_pixel_ms"] for row in selected],
                dtype=float,
            )
            net_rate = np.asarray(
                [row["net_rate_per_pixel_ms"] for row in selected],
                dtype=float,
            )

            def stats(values: np.ndarray) -> tuple[float, float, float]:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                cv = (
                    100.0 * std / abs(mean)
                    if len(values) > 1 and abs(mean) > 1e-15
                    else float("nan")
                )
                return mean, std, cv

            on_mean, on_std, on_cv = stats(on)
            net_mean, net_std, net_cv = stats(net)
            on_rate_mean, on_rate_std, on_rate_cv = stats(on_rate)
            net_rate_mean, net_rate_std, net_rate_cv = stats(net_rate)

            output.append(
                {
                    "experiment": experiment_name,
                    "illumination_us": illumination_us,
                    "illumination_ms": illumination_us / 1000.0,
                    "window_mode": window_mode,
                    "response_window_us": response_window_us,
                    "roi": roi_name,
                    "colour": colour,
                    "n_repeats": len(selected),
                    "on_mean_density": on_mean,
                    "on_std_density": on_std,
                    "on_cv_pct": on_cv,
                    "net_mean_density": net_mean,
                    "net_std_density": net_std,
                    "net_cv_pct": net_cv,
                    "on_rate_mean_per_pixel_ms": on_rate_mean,
                    "on_rate_std_per_pixel_ms": on_rate_std,
                    "on_rate_cv_pct": on_rate_cv,
                    "net_rate_mean_per_pixel_ms": net_rate_mean,
                    "net_rate_std_per_pixel_ms": net_rate_std,
                    "net_rate_cv_pct": net_rate_cv,
                }
            )

    return output


def vector_from_summary(row: dict) -> np.ndarray:
    return np.asarray(
        [row["vector_R"], row["vector_G"], row["vector_B"]],
        dtype=np.float64,
    )


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def calculate_roi_separability(response_rows: list[dict]) -> list[dict]:
    """Pairwise ROI distance inside each experiment."""
    output = []
    keys = sorted(
        {
            (
                row["experiment"],
                row["window_mode"],
                row["estimator"],
            )
            for row in response_rows
        }
    )

    for experiment, window_mode, estimator in keys:
        rows = [
            row for row in response_rows
            if row["experiment"] == experiment
            and row["window_mode"] == window_mode
            and row["estimator"] == estimator
        ]
        by_roi = {row["roi"]: row for row in rows}

        for roi_a, roi_b in combinations(sorted(by_roi), 2):
            va = vector_from_summary(by_roi[roi_a])
            vb = vector_from_summary(by_roi[roi_b])
            output.append(
                {
                    "experiment": experiment,
                    "illumination_us": int(by_roi[roi_a]["illumination_us"]),
                    "window_mode": window_mode,
                    "estimator": estimator,
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                    "euclidean_distance": euclidean_distance(va, vb),
                    "cosine_similarity": cosine_similarity(va, vb),
                }
            )

    return output


def calculate_cross_duration_robustness(response_rows: list[dict]) -> list[dict]:
    """Compare RGB vectors between every pair of illumination durations."""
    output = []

    keys = sorted(
        {
            (row["window_mode"], row["estimator"], row["roi"])
            for row in response_rows
        }
    )

    for window_mode, estimator, roi in keys:
        rows = sorted(
            [
                row for row in response_rows
                if row["window_mode"] == window_mode
                and row["estimator"] == estimator
                and row["roi"] == roi
            ],
            key=lambda row: int(row["illumination_us"]),
        )

        for row_a, row_b in combinations(rows, 2):
            va = vector_from_summary(row_a)
            vb = vector_from_summary(row_b)
            output.append(
                {
                    "window_mode": window_mode,
                    "estimator": estimator,
                    "roi": roi,
                    "experiment_a": row_a["experiment"],
                    "illumination_a_us": int(row_a["illumination_us"]),
                    "experiment_b": row_b["experiment"],
                    "illumination_b_us": int(row_b["illumination_us"]),
                    "euclidean_distance": euclidean_distance(va, vb),
                    "cosine_similarity": cosine_similarity(va, vb),
                }
            )

    return output


# =============================================================================
# Plotting
# =============================================================================

def selected_rois(response_rows: list[dict]) -> list[str]:
    rois = sorted({row["roi"] for row in response_rows})
    return rois if PLOT_ALL_ROIS else rois[:1]


def plot_rgb_vs_duration(
    response_rows: list[dict],
    output_dir: Path,
    window_mode: str,
    estimator: str,
) -> None:
    """Plot normalised R/G/B composition against illumination duration."""
    for roi in selected_rois(response_rows):
        rows = sorted(
            [
                row for row in response_rows
                if row["window_mode"] == window_mode
                and row["estimator"] == estimator
                and row["roi"] == roi
            ],
            key=lambda row: int(row["illumination_us"]),
        )
        if len(rows) < 2:
            continue

        x_ms = np.asarray([row["illumination_ms"] for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        for channel in ("R", "G", "B"):
            y = [row[f"vector_{channel}"] for row in rows]
            ax.plot(
                x_ms,
                y,
                marker="o",
                linewidth=1.8,
                label=channel,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Illumination duration (ms, log scale)")
        ax.set_ylabel("Normalised RGB response")
        ax.set_ylim(0, 1)
        ax.set_title(f"{roi}: {estimator} | {window_mode}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3)

        path = output_dir / f"rgb_vs_duration_{roi}_{window_mode}_{estimator}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_event_rate_vs_duration(
    measurements: list[dict],
    output_dir: Path,
    window_mode: str,
) -> None:
    """Compare time-normalised ON event rates across illumination durations."""
    for roi in sorted({row["roi"] for row in measurements}):
        if not PLOT_ALL_ROIS and roi != "ROI_1":
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

        for colour_name in COLOURS:
            points = []
            for illumination_us in sorted(
                {
                    int(row["illumination_us"])
                    for row in measurements
                    if row["window_mode"] == window_mode
                }
            ):
                values = [
                    float(row["on_rate_per_pixel_ms"])
                    for row in measurements
                    if row["window_mode"] == window_mode
                    and row["roi"] == roi
                    and row["colour"] == colour_name
                    and int(row["illumination_us"]) == illumination_us
                ]
                if values:
                    points.append(
                        (
                            illumination_us / 1000.0,
                            float(np.mean(values)),
                            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        )
                    )

            if points:
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                error = [p[2] for p in points]
                ax.errorbar(
                    x,
                    y,
                    yerr=error,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=colour_name,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Illumination duration (ms, log scale)")
        ax.set_ylabel("ON event rate / ROI pixel / ms")
        ax.set_title(f"{roi}: time-normalised ON response | {window_mode}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3)

        path = output_dir / f"event_rate_vs_duration_{roi}_{window_mode}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_repeatability_vs_duration(
    repeatability_rows: list[dict],
    output_dir: Path,
    window_mode: str,
) -> None:
    """Plot ON-response coefficient of variation for R/G/B."""
    for roi in sorted({row["roi"] for row in repeatability_rows}):
        if not PLOT_ALL_ROIS and roi != "ROI_1":
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

        for colour_name in COLOURS:
            rows = sorted(
                [
                    row for row in repeatability_rows
                    if row["window_mode"] == window_mode
                    and row["roi"] == roi
                    and row["colour"] == colour_name
                    and np.isfinite(float(row["on_cv_pct"]))
                ],
                key=lambda row: int(row["illumination_us"]),
            )
            if not rows:
                continue

            ax.plot(
                [row["illumination_ms"] for row in rows],
                [row["on_cv_pct"] for row in rows],
                marker="o",
                linewidth=1.5,
                label=colour_name,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Illumination duration (ms, log scale)")
        ax.set_ylabel("ON-response CV (%)")
        ax.set_title(f"{roi}: repeatability | {window_mode}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3)

        path = output_dir / f"repeatability_vs_duration_{roi}_{window_mode}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_distance_from_1ms(
    response_rows: list[dict],
    output_dir: Path,
    window_mode: str,
    estimator: str,
) -> None:
    """Plot distance from the r3/1-ms response vector."""
    baseline_name = "r3_1ms"

    for roi in selected_rois(response_rows):
        rows = [
            row for row in response_rows
            if row["window_mode"] == window_mode
            and row["estimator"] == estimator
            and row["roi"] == roi
        ]
        baseline = next(
            (row for row in rows if row["experiment"] == baseline_name),
            None,
        )
        if baseline is None:
            continue

        vb = vector_from_summary(baseline)
        rows = sorted(rows, key=lambda row: int(row["illumination_us"]))

        x = [row["illumination_ms"] for row in rows]
        y = [
            euclidean_distance(vector_from_summary(row), vb)
            for row in rows
        ]

        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        ax.plot(x, y, marker="o", linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Illumination duration (ms, log scale)")
        ax.set_ylabel("Euclidean distance from 1-ms RGB vector")
        ax.set_title(f"{roi}: cross-duration change | {window_mode} | {estimator}")
        ax.grid(alpha=0.25)

        path = output_dir / f"distance_from_1ms_{roi}_{window_mode}_{estimator}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)


# =============================================================================
# Per-recording analysis
# =============================================================================

def analyse_recording(
    experiment: Experiment,
    recording: dict,
    detection,
    polygons: dict[str, np.ndarray],
    roi_masks: dict[str, np.ndarray],
    roi_areas: dict[str, int],
    experiment_output: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run FULL and FIXED_500US quantitative response measurements."""
    all_measurements: list[dict] = []
    all_summaries: list[dict] = []
    all_repeatability: list[dict] = []

    modes = (
        ("FULL", int(experiment.illumination_us)),
        ("FIXED_500US", int(FIXED_RESPONSE_WINDOW_US)),
    )

    for window_mode, response_window_us in modes:
        timing = make_timing_config(experiment, response_window_us)
        mode_dir = experiment_output / window_mode.lower()
        mode_dir.mkdir(parents=True, exist_ok=True)

        rows = count_roi_flash_responses_chunked(
            experiment.data_file,
            recording["read_start_position"],
            recording["actual_start_timestamp"],
            recording["actual_read_start_us"],
            detection.onsets_us,
            roi_masks,
            roi_areas,
            recording["width"],
            recording["height"],
            timing,
            chunk_events=ROI_CONFIG_AUTO.chunk_events,
        )
        rows = add_derived_measurements(rows, experiment, window_mode)
        write_csv(mode_dir / "measurements.csv", rows)

        summaries = []
        for estimator in ESTIMATORS:
            summaries.extend(calculate_response_summaries(rows, estimator))
        write_csv(mode_dir / "response_estimators.csv", summaries)

        repeatability = calculate_repeatability(rows)
        write_csv(mode_dir / "repeatability.csv", repeatability)

        all_measurements.extend(rows)
        all_summaries.extend(summaries)
        all_repeatability.extend(repeatability)

    # Save the exact common ROI coordinates into every experiment folder.
    (experiment_output / "roi_polygons.json").write_text(
        json.dumps(
            {
                name: np.asarray(points, dtype=int).tolist()
                for name, points in polygons.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return all_measurements, all_summaries, all_repeatability


# =============================================================================
# Summary text
# =============================================================================

def write_summary(
    output_dir: Path,
    response_rows: list[dict],
    robustness_rows: list[dict],
) -> None:
    """Create a short machine-generated interpretation guide."""
    lines = [
        "ILLUMINATION-DURATION EXPERIMENT",
        "=" * 40,
        "",
        "Conditions:",
    ]
    for experiment in sorted(EXPERIMENTS, key=lambda e: e.illumination_us):
        lines.append(
            f"- {experiment.name}: illumination={experiment.illumination_us} us, "
            f"dark={experiment.dark_us} us, period={experiment.period_us} us"
        )

    lines += [
        "",
        "Primary interpretation:",
        "- FULL compares each recording using its complete illumination duration.",
        "- FIXED_500US compares the same first 500 us after every predicted onset.",
        "- ON_PEAK is the original baseline method.",
        "- ON_MEDIAN is a more robust repeated-cycle estimator.",
        "- NET_POS_MEDIAN uses max(ON-OFF, 0) before median aggregation.",
        "- Time-normalised rates use events / ROI pixel / ms.",
        "",
        "Cross-duration robustness:",
        "A smaller Euclidean RGB-vector distance and a cosine similarity closer "
        "to 1 indicate greater stability across illumination durations.",
        "",
    ]

    # Give ROI_1 ON_PEAK full-window vectors in a compact textual table.
    lines.append("ROI_1 FULL ON_PEAK response vectors:")
    rows = sorted(
        [
            row for row in response_rows
            if row["roi"] == "ROI_1"
            and row["window_mode"] == "FULL"
            and row["estimator"] == "ON_PEAK"
        ],
        key=lambda row: int(row["illumination_us"]),
    )
    for row in rows:
        lines.append(
            f"- {row['experiment']}: "
            f"[R={row['vector_R']:.4f}, "
            f"G={row['vector_G']:.4f}, "
            f"B={row['vector_B']:.4f}]"
        )

    # Mean pairwise distance for a few key comparisons.
    lines += ["", "Mean pairwise RGB-vector distance across durations:"]
    for window_mode, estimator in (
        ("FULL", "ON_PEAK"),
        ("FULL", "ON_MEDIAN"),
        ("FIXED_500US", "ON_PEAK"),
        ("FIXED_500US", "ON_MEDIAN"),
        ("FIXED_500US", "NET_POS_MEDIAN"),
    ):
        selected = [
            row["euclidean_distance"]
            for row in robustness_rows
            if row["roi"] == "ROI_1"
            and row["window_mode"] == window_mode
            and row["estimator"] == estimator
        ]
        if selected:
            lines.append(
                f"- {window_mode} / {estimator}: "
                f"{float(np.mean(selected)):.5f}"
            )

    (output_dir / "experiment_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    experiment_map = {experiment.name: experiment for experiment in EXPERIMENTS}
    if ROI_REFERENCE_EXPERIMENT not in experiment_map:
        raise ValueError(
            f"ROI_REFERENCE_EXPERIMENT={ROI_REFERENCE_EXPERIMENT!r} "
            "is not present in EXPERIMENTS."
        )

    if len(experiment_map) != len(EXPERIMENTS):
        raise ValueError("Experiment names must be unique.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    config_payload = {
        "fixed_response_window_us": FIXED_RESPONSE_WINDOW_US,
        "num_rgb_repeats": NUM_RGB_REPEATS,
        "roi_reference_experiment": ROI_REFERENCE_EXPERIMENT,
        "same_roi_polygons_reused": True,
        "experiments": [
            {
                "name": e.name,
                "data_file": str(e.data_file),
                "illumination_us": e.illumination_us,
                "dark_us": e.dark_us,
                "stage_period_us": e.period_us,
            }
            for e in EXPERIMENTS
        ],
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # 1. Open all recordings and verify the same sensor resolution.
    # ---------------------------------------------------------------------
    recordings: dict[str, dict] = {}
    for experiment in EXPERIMENTS:
        recordings[experiment.name] = open_recording(experiment)

    resolutions = {
        (recording["width"], recording["height"])
        for recording in recordings.values()
    }
    if len(resolutions) != 1:
        raise ValueError(
            f"Recordings do not share one sensor resolution: {sorted(resolutions)}"
        )

    # ---------------------------------------------------------------------
    # 2. Detect timing independently in every recording.
    # ---------------------------------------------------------------------
    detections = {}
    timing_configs = {}
    for experiment in EXPERIMENTS:
        experiment_output = output_dir / experiment.name
        experiment_output.mkdir(parents=True, exist_ok=True)
        detection, timing = detect_timing(
            experiment,
            recordings[experiment.name],
            experiment_output,
        )
        detections[experiment.name] = detection
        timing_configs[experiment.name] = timing

    # ---------------------------------------------------------------------
    # 3. Draw/load ROIs once from the reference recording.
    # ---------------------------------------------------------------------
    ref_experiment = experiment_map[ROI_REFERENCE_EXPERIMENT]
    ref_recording = recordings[ROI_REFERENCE_EXPERIMENT]
    ref_detection = detections[ROI_REFERENCE_EXPERIMENT]

    roi_definitions, polygons, roi_masks, roi_areas = select_or_load_common_rois(
        ref_experiment,
        ref_recording,
        ref_detection,
        output_dir,
    )

    # Save an explicit copy of the common ROI file in this experiment output.
    save_polygons(
        output_dir / "common_roi_polygons.json",
        polygons,
        ref_recording["width"],
        ref_recording["height"],
    )

    # ---------------------------------------------------------------------
    # 4. Visual alignment check in every recording using the SAME ROIs.
    # ---------------------------------------------------------------------
    for experiment in EXPERIMENTS:
        make_alignment_check(
            experiment,
            recordings[experiment.name],
            detections[experiment.name],
            polygons,
            roi_masks,
            output_dir / experiment.name / "roi_alignment_check.png",
        )

    print(
        "\nIMPORTANT: inspect each roi_alignment_check.png before interpreting "
        "the quantitative comparison. If the scene moved, delete "
        f"{ROI_POLYGON_FILE.name}, realign the recordings, and redraw ROIs."
    )

    # ---------------------------------------------------------------------
    # 5. FULL and FIXED_500US measurements for every recording.
    # ---------------------------------------------------------------------
    all_measurements: list[dict] = []
    all_response_rows: list[dict] = []
    all_repeatability: list[dict] = []

    for experiment in EXPERIMENTS:
        measurements, summaries, repeatability = analyse_recording(
            experiment,
            recordings[experiment.name],
            detections[experiment.name],
            polygons,
            roi_masks,
            roi_areas,
            output_dir / experiment.name,
        )
        all_measurements.extend(measurements)
        all_response_rows.extend(summaries)
        all_repeatability.extend(repeatability)

    # ---------------------------------------------------------------------
    # 6. Combined quantitative outputs.
    # ---------------------------------------------------------------------
    write_csv(output_dir / "all_measurements.csv", all_measurements)
    write_csv(output_dir / "duration_response_comparison.csv", all_response_rows)
    write_csv(output_dir / "duration_repeatability.csv", all_repeatability)

    separability = calculate_roi_separability(all_response_rows)
    robustness = calculate_cross_duration_robustness(all_response_rows)
    write_csv(output_dir / "roi_separability.csv", separability)
    write_csv(output_dir / "cross_duration_robustness.csv", robustness)

    # ---------------------------------------------------------------------
    # 7. Key plots.
    # ---------------------------------------------------------------------
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for mode in ("FULL", "FIXED_500US"):
        plot_event_rate_vs_duration(
            all_measurements,
            plot_dir,
            mode,
        )
        plot_repeatability_vs_duration(
            all_repeatability,
            plot_dir,
            mode,
        )

    for mode, estimator in (
        ("FULL", "ON_PEAK"),
        ("FULL", "ON_MEDIAN"),
        ("FIXED_500US", "ON_PEAK"),
        ("FIXED_500US", "ON_MEDIAN"),
        ("FIXED_500US", "NET_POS_MEDIAN"),
    ):
        plot_rgb_vs_duration(
            all_response_rows,
            plot_dir,
            mode,
            estimator,
        )
        plot_distance_from_1ms(
            all_response_rows,
            plot_dir,
            mode,
            estimator,
        )

    write_summary(
        output_dir,
        all_response_rows,
        robustness,
    )

    print("\nDuration experiment complete.")
    print(f"Results: {output_dir}")
    print("\nMost important files:")
    print("  duration_response_comparison.csv")
    print("  duration_repeatability.csv")
    print("  cross_duration_robustness.csv")
    print("  roi_separability.csv")
    print("  experiment_summary.txt")
    print("  plots/")
    print("\nInterpret FULL and FIXED_500US together:")
    print("  - FULL reveals total temporal/PWM accumulation effects.")
    print("  - FIXED_500US tests whether the early response is stable across files.")
    print("  - ON_PEAK is the original baseline.")
    print("  - ON_MEDIAN / NET_POS_MEDIAN test a more robust estimator.")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
