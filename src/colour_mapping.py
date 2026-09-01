"""ON-peak colour-response mapping and pseudo-colour visualization.

The project does not claim calibrated RGB recovery.  The colour feature is the
same ON_PEAK rule used by the original working script:
    - for each ROI, take the maximum ON density across repeats for R, G and B,
    - normalise [R_peak, G_peak, B_peak] into a relative response vector,
    - map that vector to active event pixels for pseudo-colour display.

An optional symmetric activity-based brightness multiplier is display-only and
is kept separate from the exported response vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dat_decoder import DisplayEventReader
from roi_analysis import event_display_layers, normalized_vector


@dataclass(frozen=True)
class ColourConfig:
    minimum_colour_response_events: float = 1.0
    minimum_colour_response_density: float = 1e-6
    on_peak_display_brightness: float = 200.0
    display_desaturation_fraction: float = 0.12


@dataclass(frozen=True)
class VideoConfig:
    start_after_read_us: int | None = None
    duration_us: int = 500_000
    frame_window_us: int = 500
    frame_step_us: int = 500
    persistence_us: int = 20_000
    display_mode: str = "TOTAL"
    event_point_size: int = 1
    event_clip_value: float = 1.0
    show_events_outside_rois: bool = True
    background_event_brightness: float = 0.25
    colour_blend: float = 1.0
    off_event_brightness: float = 0.20
    on_event_brightness: float = 1.00
    draw_roi_outlines: bool = False
    gif_fps: float = 15.0
    gif_max_frames: int = 90
    gif_loop: int = 0
    chunk_events: int = 1_000_000


def build_roi_colours(
    rows: list[dict],
    brightness_results: dict[str, dict],
    config: ColourConfig = ColourConfig(),
) -> tuple[dict[str, np.ndarray | None], list[dict], list[dict]]:
    """Build ON_PEAK response vectors and display BGR colours.

    Returns
    -------
    roi_colours:
        Mapping from ROI name to BGR display colour (or ``None`` if too weak).
    peak_summaries:
        One ON_PEAK summary row per ROI for CSV/plots.
    colour_diagnostics:
        Per-repeat normalized ON responses.
    """
    roi_names = sorted({row["roi"] for row in rows})
    repeats = sorted({int(row["repeat"]) for row in rows})
    colour_names = ("R", "G", "B")
    roi_colours: dict[str, np.ndarray | None] = {}
    peak_summaries: list[dict] = []
    colour_diagnostics: list[dict] = []

    for roi_name in roi_names:
        repeat_matrix = np.zeros((len(repeats), 3), dtype=np.float64)
        count_matrix = np.zeros_like(repeat_matrix)

        for repeat_index, repeat in enumerate(repeats):
            for colour_index, colour in enumerate(colour_names):
                item = next(
                    row
                    for row in rows
                    if row["roi"] == roi_name
                    and int(row["repeat"]) == repeat
                    and row["colour"] == colour
                )
                repeat_matrix[repeat_index, colour_index] = float(item["on_density"])
                count_matrix[repeat_index, colour_index] = float(item["on"])

        peak_indices = np.argmax(repeat_matrix, axis=0)
        peak_densities = np.max(repeat_matrix, axis=0)
        peak_counts = count_matrix[peak_indices, np.arange(3)]
        response_vector = normalized_vector(peak_densities)
        repeat_stds = (
            np.std(repeat_matrix, axis=0, ddof=1)
            if len(repeats) > 1
            else np.zeros(3, dtype=np.float64)
        )

        brightness = brightness_results[roi_name]
        summary = {
            "roi": roi_name,
            "metric": "ON_PEAK",
            "peak_R_count": float(peak_counts[0]),
            "peak_G_count": float(peak_counts[1]),
            "peak_B_count": float(peak_counts[2]),
            "peak_R_density": float(peak_densities[0]),
            "peak_G_density": float(peak_densities[1]),
            "peak_B_density": float(peak_densities[2]),
            "vector_R": float(response_vector[0]),
            "vector_G": float(response_vector[1]),
            "vector_B": float(response_vector[2]),
            "repeat_std_R": float(repeat_stds[0]),
            "repeat_std_G": float(repeat_stds[1]),
            "repeat_std_B": float(repeat_stds[2]),
            "peak_repeat_R": int(repeats[int(peak_indices[0])]),
            "peak_repeat_G": int(repeats[int(peak_indices[1])]),
            "peak_repeat_B": int(repeats[int(peak_indices[2])]),
            "raw_brightness_ratio": float(brightness["raw_brightness_ratio"]),
            "display_brightness_multiplier": float(
                brightness["display_brightness_multiplier"]
            ),
        }
        peak_summaries.append(summary)

        for repeat_index, repeat in enumerate(repeats):
            repeat_vector = normalized_vector(repeat_matrix[repeat_index])
            colour_diagnostics.append(
                {
                    "roi": roi_name,
                    "repeat": repeat,
                    "raw_R_density": float(repeat_matrix[repeat_index, 0]),
                    "raw_G_density": float(repeat_matrix[repeat_index, 1]),
                    "raw_B_density": float(repeat_matrix[repeat_index, 2]),
                    "normalised_R": float(repeat_vector[0]),
                    "normalised_G": float(repeat_vector[1]),
                    "normalised_B": float(repeat_vector[2]),
                }
            )

        if (
            float(np.sum(peak_counts)) < config.minimum_colour_response_events
            or float(np.sum(peak_densities)) < config.minimum_colour_response_density
            or np.max(response_vector) <= 0
        ):
            roi_colours[roi_name] = None
            summary.update(
                {
                    "base_display_R": None,
                    "base_display_G": None,
                    "base_display_B": None,
                    "final_display_R": None,
                    "final_display_G": None,
                    "final_display_B": None,
                }
            )
            continue

        display_unit = response_vector / np.max(response_vector)
        base_display_rgb = np.clip(
            display_unit * float(np.clip(config.on_peak_display_brightness, 0, 255)),
            0,
            255,
        )

        # Display-only desaturation.  The exported ON_PEAK vector is untouched.
        luminance = float(np.dot(base_display_rgb, [0.2126, 0.7152, 0.0722]))
        base_display_rgb = (
            (1.0 - config.display_desaturation_fraction) * base_display_rgb
            + config.display_desaturation_fraction * luminance
        )

        multiplier = float(brightness["display_brightness_multiplier"])
        final_rgb = np.clip(base_display_rgb * multiplier, 0, 255)
        roi_colours[roi_name] = final_rgb[::-1].astype(np.float32)  # RGB -> BGR
        summary.update(
            {
                "base_display_R": float(base_display_rgb[0]),
                "base_display_G": float(base_display_rgb[1]),
                "base_display_B": float(base_display_rgb[2]),
                "final_display_R": float(final_rgb[0]),
                "final_display_G": float(final_rgb[1]),
                "final_display_B": float(final_rgb[2]),
            }
        )

    return roi_colours, peak_summaries, colour_diagnostics


def save_rgb_response_figure(
    peak_summaries: list[dict],
    output_path: str | Path,
) -> None:
    """Save ON_PEAK response vectors and final display swatches."""
    roi_names = [row["roi"] for row in peak_summaries]
    vectors = np.asarray(
        [[row["vector_R"], row["vector_G"], row["vector_B"]] for row in peak_summaries],
        dtype=np.float64,
    )
    x = np.arange(len(roi_names), dtype=float)
    width = 0.24

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(max(9.0, 1.45 * len(roi_names) + 5.0), 7.0),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )

    for channel, offset, colour in zip(
        range(3), (-width, 0.0, width), ("#d62728", "#2ca02c", "#1f77b4")
    ):
        label = ("R", "G", "B")[channel]
        axes[0].bar(
            x + offset,
            vectors[:, channel],
            width,
            color=colour,
            alpha=0.88,
            label=label,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(roi_names)
    axes[0].set_ylabel("Normalised ON_PEAK response")
    axes[0].set_ylim(0, max(1.0, float(np.max(vectors)) * 1.12))
    axes[0].set_title("Relative RGB event-response vector")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=3, frameon=False)

    swatches = np.zeros((1, len(roi_names), 3), dtype=np.float32)
    for index, row in enumerate(peak_summaries):
        final = [row.get(f"final_display_{channel}") for channel in "RGB"]
        if all(value is not None for value in final):
            swatches[0, index] = np.asarray(final, dtype=np.float32)
        else:
            swatches[0, index] = np.array([127, 127, 127], dtype=np.float32)

    axes[1].imshow(np.clip(swatches / 255.0, 0, 1), aspect="auto", interpolation="nearest")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(roi_names)
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Pseudo-colour"])
    axes[1].set_title("Final display colours (not calibrated reflectance RGB)")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def colour_event_display_frame(
    frame_on: np.ndarray,
    frame_off: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    roi_colours: dict[str, np.ndarray | None],
    config: VideoConfig,
) -> np.ndarray:
    """Map ROI colours only onto active event pixels."""
    gray, active, strength = event_display_layers(
        frame_on,
        frame_off,
        config.display_mode,
        config.event_clip_value,
        config.event_point_size,
    )
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    union = np.zeros(gray.shape, dtype=bool)
    for mask in roi_masks.values():
        union |= mask

    # Large regions first, small nested regions later.
    ordered = sorted(
        roi_masks.items(),
        key=lambda item: np.count_nonzero(item[1]),
        reverse=True,
    )
    for roi_name, mask in ordered:
        colour = roi_colours[roi_name]
        active_roi = active & mask
        if not np.any(active_roi):
            continue
        if colour is None:
            continue

        level = strength[active_roi]
        if config.display_mode.upper() == "SIGNED":
            brightness = config.off_event_brightness + level * (
                config.on_event_brightness - config.off_event_brightness
            )
        elif config.display_mode.upper() == "OFF":
            brightness = level * config.off_event_brightness
        else:
            brightness = level * config.on_event_brightness

        recovered = colour[None, :] * brightness[:, None]
        original = np.repeat(gray[active_roi, None], 3, axis=1).astype(np.float32)
        frame[active_roi] = np.clip(
            (1 - config.colour_blend) * original + config.colour_blend * recovered,
            0,
            255,
        ).astype(np.uint8)

    if config.show_events_outside_rois:
        if config.display_mode.upper() == "SIGNED":
            background_gray = np.clip(
                127.0
                + (gray.astype(np.float32) - 127.0)
                * float(config.background_event_brightness),
                0,
                255,
            ).astype(np.uint8)
        else:
            background_gray = np.clip(
                gray.astype(np.float32) * float(config.background_event_brightness),
                0,
                255,
            ).astype(np.uint8)
        background_bgr = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
        frame[~union] = background_bgr[~union]
    else:
        frame[~union] = 127 if config.display_mode.upper() == "SIGNED" else 0

    return frame


def create_pseudo_colour_gif(
    file_path: str | Path,
    read_start_position: int,
    actual_start_timestamp: int,
    actual_read_start_us: int,
    first_onset_us: int,
    width: int,
    height: int,
    polygons: dict[str, np.ndarray],
    roi_masks: dict[str, np.ndarray],
    roi_colours: dict[str, np.ndarray | None],
    output_path: str | Path,
    config: VideoConfig = VideoConfig(),
) -> None:
    """Generate a bounded-size pseudo-colour GIF for GitHub/demo use."""
    video_start_offset = (
        int(first_onset_us)
        if config.start_after_read_us is None
        else int(config.start_after_read_us)
    )
    if video_start_offset < 0:
        raise ValueError("Video/GIF start offset cannot be negative.")
    start_timestamp = int(actual_start_timestamp + video_start_offset)

    with DisplayEventReader(
        file_path,
        read_start_position,
        width,
        height,
        chunk_events=config.chunk_events,
    ) as reader:
        duration = min(
            int(config.duration_us), reader.last_timestamp - start_timestamp + 1
        )
        if duration <= 0:
            raise ValueError("GIF start lies beyond the end of the DAT recording.")

        source_frame_count = (
            duration + int(config.frame_step_us) - 1
        ) // int(config.frame_step_us)
        sample_stride = max(
            1,
            int(math.ceil(source_frame_count / int(config.gif_max_frames))),
        )
        selected_frames = range(0, source_frame_count, sample_stride)
        frame_duration_ms = max(1, int(round(1000.0 / config.gif_fps)))
        gif_frames: list[Image.Image] = []

        for source_index in selected_frames:
            frame_start = start_timestamp + source_index * int(config.frame_step_us)
            display_start = max(
                int(actual_start_timestamp),
                frame_start - int(config.persistence_us),
            )
            display_end = min(
                reader.last_timestamp + 1,
                frame_start + int(config.frame_window_us),
            )
            on, off, _ = reader.window_counts(display_start, display_end)
            frame = colour_event_display_frame(
                on, off, roi_masks, roi_colours, config
            )

            if config.draw_roi_outlines:
                for name, polygon in polygons.items():
                    colour = roi_colours.get(name)
                    if colour is not None:
                        cv2.polylines(
                            frame,
                            [polygon],
                            True,
                            tuple(int(value) for value in colour),
                            1,
                        )

            relative_ms = (
                actual_read_start_us + frame_start - actual_start_timestamp
            ) / 1000.0
            label = f"Pseudo-colour event result | t={relative_ms:.3f} ms"
            cv2.rectangle(frame, (0, 0), (min(width - 1, 355), 24), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (7, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(
                Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE)
            )

        if not gif_frames:
            raise RuntimeError("No frames were generated for the pseudo-colour GIF.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration_ms,
            loop=int(config.gif_loop),
            optimize=False,
            disposal=2,
        )


def save_peak_summary_csv(path: str | Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
