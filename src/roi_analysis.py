"""ROI selection and quantitative event-response analysis.

This module handles the spatial part of the experiment:
    1. build a clear event reference image,
    2. interactively draw polygonal ROIs,
    3. make nested ROIs mutually exclusive when requested,
    4. count ON/OFF events in fixed RGB response windows,
    5. export ROI-response and repeatability summaries.

Display accumulation is intentionally separated from the quantitative response
windows so a longer reference frame does not alter the colour measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dat_decoder import DisplayEventReader, iter_event_chunks
from temporal_alignment import TimingConfig, TimingDetectionError


@dataclass(frozen=True)
class ROIConfig:
    num_rois: int | None = None
    nested_roi_mode: str = "EXCLUDE_CONTAINED"
    reference_window_us: int = 50_000
    reference_step_us: int = 10_000
    reference_search_duration_us: int = 300_000
    reference_start_after_read_us: int | None = None
    reference_clip_value: float = 1.0
    reference_display_mode: str = "TOTAL"
    display_scale: float = 1.5
    event_point_size: int = 1
    automatic_reference_frame: bool = False
    window_presets_us: tuple[int, ...] = (
        100,
        500,
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
    )
    chunk_events: int = 1_000_000


@dataclass
class ROISelection:
    definitions: list[tuple[str, str, tuple[int, int, int]]]
    polygons: dict[str, np.ndarray]
    raw_masks: dict[str, np.ndarray]
    effective_masks: dict[str, np.ndarray]
    raw_areas: dict[str, int]
    effective_areas: dict[str, int]
    region_details: dict
    reference_details: dict


def configure_rois(num_rois: int | None) -> list[tuple[str, str, tuple[int, int, int]]]:
    count = num_rois
    while count is None:
        try:
            answer = int(input("Number of ROIs to select (positive integer): "))
            if answer > 0:
                count = answer
            else:
                print("ROI count must be greater than zero.")
        except ValueError:
            print("Please enter an integer, e.g. 1, 3, or 5.")

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("num_rois must be a positive integer or None.")

    definitions = []
    for i in range(count):
        label = (
            f"{i + 1}/{count} Select normal/reference surface"
            if i == 0
            else f"{i + 1}/{count} Select defect/test region"
        )
        definitions.append((f"ROI_{i + 1}", label, (255, 255, 255)))
    return definitions


def save_png(path: str | Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    encoded.tofile(str(path))


def event_display_layers(
    frame_on: np.ndarray,
    frame_off: np.ndarray,
    mode: str,
    clip_value: float,
    point_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grayscale display, activity mask, and normalized strength."""
    if clip_value <= 0 or point_size not in (1, 3):
        raise ValueError("clip_value must be positive and point_size must be 1 or 3.")

    mode = mode.upper()
    if mode == "SIGNED":
        signed = np.clip((frame_on - frame_off) / float(clip_value), -1, 1)
        strength = (signed + 1) / 2
        active = (frame_on + frame_off) > 0
        return (
            (strength * 255).astype(np.uint8),
            active,
            strength.astype(np.float32),
        )

    if mode == "TOTAL":
        values = frame_on + frame_off
    elif mode == "ON":
        values = frame_on
    elif mode == "OFF":
        values = frame_off
    else:
        raise ValueError("Display mode must be TOTAL, SIGNED, ON or OFF.")

    strength = np.clip(values / float(clip_value), 0, 1).astype(np.float32)
    if point_size == 3:
        strength = cv2.dilate(strength, np.ones((3, 3), np.uint8))
    return (strength * 255).astype(np.uint8), strength > 0, strength


def append_display_caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    height, width = image.shape[:2]
    caption_height = 10 + 20 * len(lines)
    display = np.full(
        (
            height + caption_height + (height + caption_height) % 2,
            width + width % 2,
            3,
        ),
        24,
        np.uint8,
    )
    display[:height, :width] = image
    for i, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (5, height + 17 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return display


def _display_window_key(window_name: str, delay_ms: int) -> int:
    key = cv2.waitKey(delay_ms) & 0xFF
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return ord("q")
    return key


def _choose_reference_by_coverage(
    reader: DisplayEventReader,
    starts: np.ndarray,
    window_us: int,
    actual_start_timestamp: int,
) -> int:
    best = 0
    best_coverage = -1
    for i, offset in enumerate(starts):
        on, off, _ = reader.window_counts(
            actual_start_timestamp + int(offset),
            actual_start_timestamp + int(offset) + window_us,
        )
        coverage = int(np.count_nonzero(on + off))
        if coverage > best_coverage:
            best = i
            best_coverage = coverage
    if best_coverage <= 0:
        raise TimingDetectionError("No useful event pixels are available for ROI drawing.")
    return best


def select_clear_reference_frame(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    actual_read_start_us: int,
    width: int,
    height: int,
    first_onset_us: int,
    config: ROIConfig,
) -> tuple[np.ndarray, int, dict]:
    """Interactively choose a display-only event accumulation for ROI drawing."""
    window_us = int(config.reference_window_us)
    clip = float(config.reference_clip_value)
    mode = config.reference_display_mode.upper()
    point_size = int(config.event_point_size)
    start_offset = max(
        0,
        int(
            first_onset_us
            if config.reference_start_after_read_us is None
            else config.reference_start_after_read_us
        ),
    )
    window_name = "ROI reference - display only; timing stays fixed"

    with DisplayEventReader(
        file_path,
        read_start_position,
        width,
        height,
        chunk_events=config.chunk_events,
    ) as reader:
        last_offset = reader.last_timestamp - int(actual_start_timestamp)
        last_start = min(
            last_offset,
            start_offset + int(config.reference_search_duration_us) - 1,
        )
        if last_start < start_offset:
            raise ValueError("ROI reference start lies beyond the DAT recording.")

        starts = np.arange(
            start_offset,
            last_start + 1,
            int(config.reference_step_us),
            dtype=np.int64,
        )
        current = _choose_reference_by_coverage(
            reader, starts, window_us, actual_start_timestamp
        )

        if not config.automatic_reference_frame:
            print(
                "ROI reference controls: A/D time, W/S window, +/- brightness, "
                "M mode, T point size, R auto-select, Enter confirm, Q cancel."
            )
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                window_name,
                int(width * config.display_scale),
                int((height + 90) * config.display_scale),
            )

        try:
            while True:
                frame_local_us = int(starts[current])
                end_local_us = min(frame_local_us + window_us, last_offset + 1)
                on, off, event_count = reader.window_counts(
                    actual_start_timestamp + frame_local_us,
                    actual_start_timestamp + end_local_us,
                )
                dot_size = 1 if mode == "SIGNED" else point_size
                gray, _, _ = event_display_layers(
                    on, off, mode, clip, dot_size
                )
                raw_coverage = int(np.count_nonzero(on + off))

                if config.automatic_reference_frame:
                    key = 13
                else:
                    display = append_display_caption(
                        gray,
                        [
                            f"Reference {current + 1}/{len(starts)} | window={window_us / 1000:g} ms | {mode}",
                            f"t={frame_local_us / 1e6:.6f}s after read | events={event_count:,} | active pixels={raw_coverage:,}",
                            "A/D:time  W/S:window  +/-:brightness  M:mode  T:dot  R:auto",
                            "Enter:draw ROIs  Q:cancel | quantitative timing unchanged",
                        ],
                    )
                    cv2.imshow(window_name, display)
                    key = _display_window_key(window_name, 0)

                if key in (13, 10, ord(" ")):
                    if not np.any(on + off):
                        print("This reference frame contains no events.")
                        continue
                    details = {
                        "start_after_read_us": frame_local_us,
                        "end_after_read_us": end_local_us,
                        "start_after_file_first_us": int(
                            actual_read_start_us + frame_local_us
                        ),
                        "requested_window_us": window_us,
                        "actual_window_us": end_local_us - frame_local_us,
                        "mode": mode,
                        "clip_value": clip,
                        "point_size": dot_size,
                        "raw_event_count": event_count,
                        "raw_active_pixels": raw_coverage,
                        "timing_and_colour_statistics_unchanged": True,
                    }
                    return gray, frame_local_us, details
                if key == ord("a"):
                    current = max(0, current - 1)
                elif key == ord("d"):
                    current = min(len(starts) - 1, current + 1)
                elif key in (ord("w"), ord("s")):
                    options = sorted(set(config.window_presets_us) | {window_us})
                    i = options.index(window_us)
                    i = min(i + 1, len(options) - 1) if key == ord("w") else max(i - 1, 0)
                    window_us = options[i]
                elif key in (ord("+"), ord("=")):
                    clip = max(0.25, clip / 2)
                elif key in (ord("-"), ord("_")):
                    clip = min(128.0, clip * 2)
                elif key == ord("m"):
                    modes = ("TOTAL", "SIGNED", "ON", "OFF")
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
                elif key == ord("t"):
                    point_size = 3 if point_size == 1 else 1
                elif key == ord("r"):
                    current = _choose_reference_by_coverage(
                        reader, starts, window_us, actual_start_timestamp
                    )
                elif key in (ord("q"), 27):
                    raise KeyboardInterrupt("ROI reference selection cancelled.")
        finally:
            if not config.automatic_reference_frame:
                cv2.destroyWindow(window_name)


def choose_polygon_rois(
    reference_gray: np.ndarray,
    roi_definitions: list[tuple[str, str, tuple[int, int, int]]],
    display_scale: float = 1.5,
) -> dict[str, np.ndarray]:
    """Interactively draw arbitrary polygonal ROIs in native sensor coordinates."""
    base = cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2BGR)
    height, width = reference_gray.shape
    completed: dict[str, np.ndarray] = {}

    print(
        "Draw each ROI: left click adds a point; right click/Z undoes; "
        "R redraws; Enter confirms; Q cancels."
    )

    for roi_name, label, colour in roi_definitions:
        points: list[tuple[int, int]] = []
        window_name = f"Draw ROI - {label}"

        def mouse_callback(event, mouse_x, mouse_y, flags, parameter):
            del flags, parameter
            if (
                event == cv2.EVENT_LBUTTONDOWN
                and 0 <= mouse_x < width
                and 0 <= mouse_y < height
            ):
                points.append((mouse_x, mouse_y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            int(width * display_scale),
            int((height + 50) * display_scale),
        )
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display = base.copy()
            for polygon in completed.values():
                cv2.polylines(display, [polygon], True, (255, 255, 255), 2)
            if points:
                polygon = np.array(points, dtype=np.int32)
                cv2.polylines(display, [polygon], len(points) >= 3, colour, 2)
                for point in points:
                    cv2.circle(display, point, 4, colour, -1)

            display = append_display_caption(
                display,
                [
                    label + " | Left:add | Right/Z:undo",
                    "Enter:confirm | R:redraw | Q:cancel | native sensor coordinates",
                ],
            )
            cv2.imshow(window_name, display)
            key = _display_window_key(window_name, 20)
            if key in (13, 10, ord(" ")):
                if len(points) >= 3:
                    completed[roi_name] = np.array(points, dtype=np.int32)
                    break
                print(f"{roi_name} requires at least three points.")
            elif key in (ord("z"), 8) and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("ROI drawing cancelled.")
        cv2.destroyWindow(window_name)

    return completed


def make_roi_masks(
    polygons: dict[str, np.ndarray], width: int, height: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    masks: dict[str, np.ndarray] = {}
    areas: dict[str, int] = {}
    for roi_name, polygon in polygons.items():
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        masks[roi_name] = mask.astype(bool)
        areas[roi_name] = int(np.count_nonzero(mask))
        if areas[roi_name] == 0:
            raise ValueError(f"Area of {roi_name} is zero.")
    return masks, areas


def prepare_analysis_roi_masks(
    raw_masks: dict[str, np.ndarray],
    roi_definitions: list[tuple[str, str, tuple[int, int, int]]],
    nested_roi_mode: str = "EXCLUDE_CONTAINED",
) -> tuple[dict[str, np.ndarray], dict[str, int], dict]:
    """Optionally subtract strictly contained child ROIs from their parents."""
    roi_names = [item[0] for item in roi_definitions]
    if set(raw_masks) != set(roi_names):
        raise ValueError("ROI masks do not match the configured ROI names.")
    if nested_roi_mode not in ("EXCLUDE_CONTAINED", "INDEPENDENT"):
        raise ValueError("nested_roi_mode must be EXCLUDE_CONTAINED or INDEPENDENT.")

    original = {name: np.asarray(raw_masks[name], dtype=bool).copy() for name in roi_names}
    contains = {name: [] for name in roi_names}
    partial_overlaps = []

    for index, name_a in enumerate(roi_names):
        area_a = int(np.count_nonzero(original[name_a]))
        for name_b in roi_names[index + 1:]:
            area_b = int(np.count_nonzero(original[name_b]))
            shared = int(np.count_nonzero(original[name_a] & original[name_b]))
            if shared == 0:
                continue
            if area_a == area_b and shared == area_a:
                raise ValueError(f"{name_a} and {name_b} have identical masks.")
            if shared == area_b and area_b < area_a:
                contains[name_a].append(name_b)
            elif shared == area_a and area_a < area_b:
                contains[name_b].append(name_a)
            else:
                partial_overlaps.append(
                    {"roi_a": name_a, "roi_b": name_b, "shared_pixels": shared}
                )

    effective = {name: mask.copy() for name, mask in original.items()}
    if nested_roi_mode == "EXCLUDE_CONTAINED":
        for parent, children in contains.items():
            for child in children:
                effective[parent] &= ~original[child]

    details = {}
    effective_areas = {}
    for name in roi_names:
        original_area = int(np.count_nonzero(original[name]))
        effective_area = int(np.count_nonzero(effective[name]))
        if effective_area == 0:
            raise ValueError(f"{name} becomes empty after nested-ROI exclusion.")
        effective_areas[name] = effective_area
        details[name] = {
            "original_polygon_area_pixels": original_area,
            "effective_statistical_area_pixels": effective_area,
            "excluded_contained_rois": (
                contains[name] if nested_roi_mode == "EXCLUDE_CONTAINED" else []
            ),
            "contained_rois_detected": contains[name],
        }

    region_details = {
        "mode": nested_roi_mode,
        "regions": details,
        "partial_overlaps_not_modified": partial_overlaps,
        "statistical_masks_are_mutually_exclusive_for_strict_nesting": (
            nested_roi_mode == "EXCLUDE_CONTAINED"
        ),
    }
    return effective, effective_areas, region_details


def save_effective_roi_mask_figure(
    reference_gray: np.ndarray,
    polygons: dict[str, np.ndarray],
    effective_masks: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    base = cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(base)
    palette = [
        (0, 255, 255),
        (255, 80, 80),
        (80, 255, 80),
        (255, 80, 255),
        (80, 160, 255),
        (255, 255, 80),
    ]
    for index, name in enumerate(effective_masks):
        overlay[effective_masks[name]] = palette[index % len(palette)]
    visual = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
    for index, (name, polygon) in enumerate(polygons.items()):
        colour = palette[index % len(palette)]
        cv2.polylines(visual, [polygon], True, colour, 2)
        cv2.putText(
            visual,
            name,
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )
    save_png(output_path, visual)


def count_roi_flash_responses_chunked(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    actual_read_start_us: int,
    onsets_us: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    roi_areas: dict[str, int],
    width: int,
    height: int,
    timing_config: TimingConfig,
    chunk_events: int = 1_000_000,
) -> list[dict]:
    """Count ON/OFF events inside every fixed RGB response window."""
    roi_names = list(roi_masks)
    accumulators = {
        (flash_index, roi_name): [0, 0]
        for flash_index in range(len(onsets_us))
        for roi_name in roi_names
    }

    final_timestamp = actual_start_timestamp + int(
        onsets_us[-1]
        + timing_config.response_delay_us
        + timing_config.response_window_us
    )
    processed = 0

    for timestamps, x, y, polarity in iter_event_chunks(
        file_path,
        read_start_position,
        final_timestamp,
        chunk_events=chunk_events,
    ):
        processed += len(timestamps)
        valid_coordinate = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        local_t = timestamps.astype(np.int64) - int(actual_start_timestamp)

        for flash_index, onset in enumerate(onsets_us):
            start = int(onset + timing_config.response_delay_us)
            stop = start + int(timing_config.response_window_us)
            left = int(np.searchsorted(local_t, start, side="left"))
            right = int(np.searchsorted(local_t, stop, side="left"))
            if right <= left:
                continue

            window_valid = valid_coordinate[left:right]
            xw = x[left:right][window_valid]
            yw = y[left:right][window_valid]
            pw = polarity[left:right][window_valid]

            for roi_name in roi_names:
                inside = roi_masks[roi_name][yw, xw]
                roi_p = pw[inside]
                accumulators[(flash_index, roi_name)][0] += int(np.count_nonzero(roi_p == 1))
                accumulators[(flash_index, roi_name)][1] += int(np.count_nonzero(roi_p == 0))

    rows = []
    for flash_index, (colour, onset) in enumerate(zip(timing_config.flash_colours, onsets_us)):
        for roi_name in roi_names:
            on_count, off_count = accumulators[(flash_index, roi_name)]
            area = roi_areas[roi_name]
            rows.append(
                {
                    "roi": roi_name,
                    "roi_area_pixels": area,
                    "repeat": flash_index // 3 + 1,
                    "colour": colour,
                    "predicted_onset_us": actual_read_start_us + int(onset),
                    "window_start_us": actual_read_start_us + int(onset + timing_config.response_delay_us),
                    "window_end_us": actual_read_start_us + int(onset + timing_config.response_delay_us + timing_config.response_window_us),
                    "response_window_us": int(timing_config.response_window_us),
                    "on": on_count,
                    "off": off_count,
                    "total": on_count + off_count,
                    "on_density": on_count / area,
                    "off_density": off_count / area,
                    "total_density": (on_count + off_count) / area,
                }
            )

    print(f"ROI response counting complete: scanned {processed:,} events.")
    return rows


def normalized_vector(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    return values / total if total > 0 else np.zeros(3, dtype=np.float64)


def calculate_resulting_vectors(rows: list[dict]) -> list[dict]:
    """Calculate mean ON/OFF/TOTAL response vectors for diagnostic export."""
    summaries = []
    roi_names = sorted({row["roi"] for row in rows})
    for roi_name in roi_names:
        roi_rows = [row for row in rows if row["roi"] == roi_name]
        area = roi_rows[0]["roi_area_pixels"]
        for metric in ("on", "off", "total"):
            means = [
                float(np.mean([row[metric] for row in roi_rows if row["colour"] == colour]))
                for colour in ("R", "G", "B")
            ]
            densities = np.asarray(means, dtype=np.float64) / area
            vector = normalized_vector(densities)
            summaries.append(
                {
                    "roi": roi_name,
                    "metric": metric.upper(),
                    "mean_R_count": means[0],
                    "mean_G_count": means[1],
                    "mean_B_count": means[2],
                    "mean_R_density": densities[0],
                    "mean_G_density": densities[1],
                    "mean_B_density": densities[2],
                    "vector_R": vector[0],
                    "vector_G": vector[1],
                    "vector_B": vector[2],
                }
            )
    return summaries


def calculate_repeatability_summary(rows: list[dict]) -> list[dict]:
    results = []
    roi_names = sorted({row["roi"] for row in rows})
    for roi_name in roi_names:
        for colour in ("R", "G", "B"):
            selected = sorted(
                [row for row in rows if row["roi"] == roi_name and row["colour"] == colour],
                key=lambda row: row["repeat"],
            )
            for metric in ("on", "off", "total"):
                density_key = metric + "_density"
                densities = np.asarray([row[density_key] for row in selected], dtype=np.float64)
                mean_density = float(np.mean(densities))
                std_density = float(np.std(densities, ddof=1)) if len(densities) > 1 else 0.0
                cv_percent = 100.0 * std_density / mean_density if mean_density > 0 else float("nan")
                results.append(
                    {
                        "roi": roi_name,
                        "colour": colour,
                        "metric": metric.upper(),
                        "mean_density": mean_density,
                        "std_density": std_density,
                        "cv_percent": cv_percent,
                    }
                )
    return results


def calculate_on_activity_brightness(
    rows: list[dict],
    reference_roi: str = "ROI_1",
    contrast_strength: float = 9.0,
    minimum_multiplier: float = 0.22,
) -> dict[str, dict]:
    """Symmetric display-only material contrast based on total ON activity."""
    roi_names = sorted({row["roi"] for row in rows})
    if reference_roi not in roi_names:
        raise ValueError(f"Reference ROI {reference_roi!r} does not exist.")

    activity = {
        name: float(sum(row["on_density"] for row in rows if row["roi"] == name))
        for name in roi_names
    }
    reference = activity[reference_roi]
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference ROI has no valid ON-event activity.")

    results = {}
    for name in roi_names:
        raw_ratio = activity[name] / reference
        if np.isfinite(raw_ratio) and raw_ratio > 0:
            log_deviation = float(abs(math.log(raw_ratio)))
            similarity = float(math.exp(-contrast_strength * log_deviation))
        else:
            log_deviation = float("inf")
            similarity = 0.0
        multiplier = float(minimum_multiplier + (1.0 - minimum_multiplier) * similarity)
        results[name] = {
            "roi": name,
            "brightness_reference_roi": reference_roi,
            "on_activity_sum_density": activity[name],
            "reference_on_activity_sum_density": reference,
            "raw_brightness_ratio": raw_ratio,
            "absolute_log_activity_deviation": log_deviation,
            "activity_similarity_to_reference": similarity,
            "display_brightness_multiplier": multiplier,
            "display_interpretation": "display-only material contrast; not calibrated brightness",
        }
    return results


def save_roi_response_figure(
    reference_gray: np.ndarray,
    polygons: dict[str, np.ndarray],
    rows: list[dict],
    output_path: str | Path,
) -> None:
    roi_names = list(polygons)
    colour_names = ("R", "G", "B")
    bar_colours = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}
    roi_palette = plt.get_cmap("tab10")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(12.0, 2.0 * len(roi_names) + 8.0), 5.8),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=True,
    )

    axes[0].imshow(reference_gray, cmap="gray", vmin=0, vmax=255)
    for index, name in enumerate(roi_names):
        polygon = np.asarray(polygons[name], dtype=np.int32)
        closed = np.vstack((polygon, polygon[0]))
        colour = roi_palette(index % 10)
        axes[0].plot(closed[:, 0], closed[:, 1], lw=2.0, color=colour)
        axes[0].annotate(
            name,
            xy=tuple(polygon[0]),
            xytext=(5, 6),
            textcoords="offset points",
            color="white",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "fc": colour, "ec": "white", "alpha": 0.88},
        )
    axes[0].set_title("Selected statistical ROIs")
    axes[0].set_axis_off()

    x = np.arange(len(roi_names), dtype=float)
    width = 0.24
    for offset, colour_name in zip((-width, 0.0, width), colour_names):
        means, errors = [], []
        for name in roi_names:
            values = np.asarray(
                [row["on_density"] for row in rows if row["roi"] == name and row["colour"] == colour_name],
                dtype=np.float64,
            )
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        axes[1].bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=bar_colours[colour_name],
            alpha=0.85,
            label=f"{colour_name} LED",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(roi_names, rotation=25, ha="right")
    axes[1].set_ylabel("Mean ON events / effective ROI pixel")
    axes[1].set_title("ROI response across RGB repetitions")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
