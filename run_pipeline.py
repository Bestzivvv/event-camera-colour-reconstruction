"""End-to-end runner for the refactored DVS colour-response project.

Edit DATA_FILE before running:
    python run_pipeline.py

Dependencies:
    python -m pip install numpy opencv-python matplotlib pillow
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys

# Allow running directly from the repository root without installing a package.
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
    calculate_on_activity_brightness,
    calculate_repeatability_summary,
    calculate_resulting_vectors,
    choose_polygon_rois,
    configure_rois,
    count_roi_flash_responses_chunked,
    make_roi_masks,
    prepare_analysis_roi_masks,
    save_effective_roi_mask_figure,
    save_png,
    save_roi_response_figure,
    select_clear_reference_frame,
    write_csv,
)
from colour_mapping import (  # noqa: E402
    ColourConfig,
    VideoConfig,
    build_roi_colours,
    create_pseudo_colour_gif,
    save_peak_summary_csv,
    save_rgb_response_figure,
)


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
DATA_FILE = Path("data/example.dat")
SKIP_BEFORE_TIME_US = 0

# These values must match the actual recording setup.
FADE_DURATION_US = 1_000
PEAK_APPLY_TIME_US = 0
DARK_INTERVAL_US = 500
END_DELAY_US = 0
MEASURED_FLASH_PERIOD_US = None

FLASH_PERIOD_US = (
    FADE_DURATION_US + PEAK_APPLY_TIME_US + DARK_INTERVAL_US
    if MEASURED_FLASH_PERIOD_US is None
    else int(MEASURED_FLASH_PERIOD_US)
)

TIMING = TimingConfig(
    num_rgb_repeats=3,
    fade_duration_us=FADE_DURATION_US,
    peak_apply_time_us=PEAK_APPLY_TIME_US,
    dark_interval_us=DARK_INTERVAL_US,
    flash_period_us=FLASH_PERIOD_US,
    cycle_extra_delay_us=END_DELAY_US,
    sync_offset_us=0,
    response_delay_us=0,
    response_window_us=FADE_DURATION_US,
    flash_detection_bin_us=max(1, min(10, FADE_DURATION_US // 20)),
    first_red_baseline_us=1_000,
    timing_min_flash_score=2.5,
    timing_min_sequence_score=5.0,
    timing_min_supported_fraction=2 / 3,
    timing_max_startup_score=3.0,
    timing_edge_guard_us=max(
        max(1, min(10, FADE_DURATION_US // 20)),
        min(50, DARK_INTERVAL_US // 5, FADE_DURATION_US // 5),
    ),
)

FLASH_SEARCH = FlashSearchConfig(
    first_red_search_duration_us=None,
    detection_block_duration_us=1_000_000,
    overview_bin_us=1_000,
    max_detection_bins=2_000_000,
    chunk_events=1_000_000,
)

ROI = ROIConfig(
    num_rois=None,
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

COLOUR = ColourConfig(
    minimum_colour_response_events=1.0,
    minimum_colour_response_density=1e-6,
    on_peak_display_brightness=200.0,
    display_desaturation_fraction=0.12,
)

VIDEO = VideoConfig(
    start_after_read_us=None,
    duration_us=500_000,
    frame_window_us=500,
    frame_step_us=500,
    persistence_us=20_000,
    display_mode="TOTAL",
    event_point_size=1,
    event_clip_value=1,
    show_events_outside_rois=True,
    background_event_brightness=0.25,
    colour_blend=1.0,
    off_event_brightness=0.20,
    on_event_brightness=1.00,
    draw_roi_outlines=False,
    gif_fps=15.0,
    gif_max_frames=90,
    gif_loop=0,
    chunk_events=1_000_000,
)

BRIGHTNESS_REFERENCE_ROI = "ROI_1"
BRIGHTNESS_CONTRAST_STRENGTH = 9.0
BRIGHTNESS_MIN_MULTIPLIER = 0.22


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"DAT file not found: {DATA_FILE}\nEdit DATA_FILE in run_pipeline.py first."
        )

    output_dir = (
        ROOT
        / "new_led_full_recording_results"
        / DATA_FILE.stem
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    header = read_dat_header(DATA_FILE)
    print_dat_header(header)
    if header.width is None or header.height is None:
        raise ValueError("DAT header is missing Width/Height.")

    target = header.file_first_timestamp + int(SKIP_BEFORE_TIME_US)
    read_start_position, actual_start_timestamp, skipped_count = (
        find_file_position_at_timestamp(DATA_FILE, header.data_start, target)
    )
    actual_read_start_us = actual_start_timestamp - header.file_first_timestamp
    width, height = int(header.width), int(header.height)

    print(
        f"Read start: {actual_read_start_us} us after the first DAT event; "
        f"skipped {skipped_count:,} events."
    )
    print(
        "R1 identity comes from the experimental start order; the event stream "
        "alone does not identify an unknown first colour."
    )

    # 1. Detect one R1 onset and lock the rest to the fixed hardware schedule.
    detection = detect_flash_onsets_chunked(
        DATA_FILE,
        read_start_position,
        actual_start_timestamp,
        actual_read_start_us,
        output_dir,
        TIMING,
        FLASH_SEARCH,
    )
    save_flash_figure(
        detection,
        actual_read_start_us,
        output_dir,
        TIMING,
        show=False,
    )

    # 2. Select a display-only event reference and draw ROIs.
    roi_definitions = configure_rois(ROI.num_rois)
    reference_gray, reference_start_us, reference_details = (
        select_clear_reference_frame(
            DATA_FILE,
            read_start_position,
            actual_start_timestamp,
            actual_read_start_us,
            width,
            height,
            first_onset_us=int(detection.onsets_us[0]),
            config=ROI,
        )
    )
    save_png(output_dir / "roi_reference.png", reference_gray)

    polygons = choose_polygon_rois(
        reference_gray,
        roi_definitions,
        display_scale=ROI.display_scale,
    )
    raw_masks, raw_areas = make_roi_masks(polygons, width, height)
    roi_masks, roi_areas, region_details = prepare_analysis_roi_masks(
        raw_masks,
        roi_definitions,
        nested_roi_mode=ROI.nested_roi_mode,
    )
    save_effective_roi_mask_figure(
        reference_gray,
        polygons,
        roi_masks,
        output_dir / "roi_effective_masks.png",
    )

    # 3. Quantitative ON/OFF response extraction from the fixed RGB windows.
    rows = count_roi_flash_responses_chunked(
        DATA_FILE,
        read_start_position,
        actual_start_timestamp,
        actual_read_start_us,
        detection.onsets_us,
        roi_masks,
        roi_areas,
        width,
        height,
        TIMING,
        chunk_events=ROI.chunk_events,
    )
    save_roi_response_figure(
        reference_gray,
        polygons,
        rows,
        output_dir / "roi_response.png",
    )

    diagnostic_vectors = calculate_resulting_vectors(rows)
    repeatability = calculate_repeatability_summary(rows)
    brightness = calculate_on_activity_brightness(
        rows,
        reference_roi=BRIGHTNESS_REFERENCE_ROI,
        contrast_strength=BRIGHTNESS_CONTRAST_STRENGTH,
        minimum_multiplier=BRIGHTNESS_MIN_MULTIPLIER,
    )

    # 4. ON_PEAK relative colour response and pseudo-colour mapping.
    roi_colours, peak_summaries, colour_diagnostics = build_roi_colours(
        rows,
        brightness,
        COLOUR,
    )
    save_rgb_response_figure(
        peak_summaries,
        output_dir / "rgb_response.png",
    )

    write_csv(output_dir / "flash_responses.csv", rows)
    write_csv(output_dir / "diagnostic_response_vectors.csv", diagnostic_vectors)
    write_csv(output_dir / "repeatability_summary.csv", repeatability)
    write_csv(output_dir / "normalised_colour_repeats.csv", colour_diagnostics)
    save_peak_summary_csv(output_dir / "response_vectors.csv", peak_summaries)

    create_pseudo_colour_gif(
        DATA_FILE,
        read_start_position,
        actual_start_timestamp,
        actual_read_start_us,
        first_onset_us=int(detection.onsets_us[0]),
        width=width,
        height=height,
        polygons=polygons,
        roi_masks=roi_masks,
        roi_colours=roi_colours,
        output_path=output_dir / "pseudo_colour_result.gif",
        config=VIDEO,
    )

    metadata = {
        "data_file": str(DATA_FILE),
        "actual_read_start_us": int(actual_read_start_us),
        "timing_detection": detection.details,
        "recording_overview": detection.recording_overview,
        "roi_reference": reference_details,
        "roi_reference_start_us": int(reference_start_us),
        "raw_roi_areas": raw_areas,
        "effective_roi_areas": roi_areas,
        "roi_region_details": region_details,
        "brightness_response": brightness,
        "polygons": {name: polygon.tolist() for name, polygon in polygons.items()},
        "interpretation": (
            "Colour is a relative ON_PEAK event-response vector. Material-contrast "
            "brightness is display-only, not calibrated RGB or absolute intensity."
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nAnalysis complete: {output_dir}")
    print("Generated core GitHub figures:")
    print("  flash_detection.png")
    print("  roi_response.png")
    print("  rgb_response.png")
    print("  pseudo_colour_result.gif")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Safe cleanup after interactive OpenCV ROI windows.
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
