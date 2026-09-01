"""Fixed-period RGB temporal alignment for the event-camera experiment.

The important design choice from the original project is preserved here:
only the first illumination onset is detected from the event stream.  All
subsequent R/G/B windows are generated from the configured hardware period;
they are not independently shifted to nearby event peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TimingConfig:
    num_rgb_repeats: int = 3
    fade_duration_us: int = 1_000
    peak_apply_time_us: int = 0
    dark_interval_us: int = 500
    flash_period_us: int = 1_500
    cycle_extra_delay_us: int = 0
    sync_offset_us: int = 0
    response_delay_us: int = 0
    response_window_us: int = 1_000
    flash_detection_bin_us: int = 10
    first_red_baseline_us: int = 1_000
    timing_min_flash_score: float = 2.5
    timing_min_sequence_score: float = 5.0
    timing_min_supported_fraction: float = 2 / 3
    timing_max_startup_score: float = 3.0
    timing_edge_guard_us: int = 50

    @property
    def flash_colours(self) -> tuple[str, ...]:
        return tuple(["R", "G", "B"] * self.num_rgb_repeats)


class TimingDetectionError(ValueError):
    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def validate_timing_config(config: TimingConfig) -> None:
    if config.num_rgb_repeats < 1:
        raise ValueError("num_rgb_repeats must be positive.")
    integer_positive = (
        config.fade_duration_us,
        config.dark_interval_us,
        config.flash_period_us,
        config.response_window_us,
        config.flash_detection_bin_us,
        config.first_red_baseline_us,
        config.timing_edge_guard_us,
    )
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
           for v in integer_positive):
        raise ValueError("Timing durations and bin widths must be positive integers.")
    if config.response_delay_us < 0 or config.cycle_extra_delay_us < 0:
        raise ValueError("Delay values must be non-negative.")
    if config.response_delay_us + config.response_window_us > config.flash_period_us:
        raise ValueError("The response window extends into the next colour stage.")
    if config.flash_detection_bin_us >= config.flash_period_us:
        raise ValueError("Detection bin must be shorter than the flash period.")
    if not 0 < config.timing_min_supported_fraction <= 1:
        raise ValueError("timing_min_supported_fraction must be in (0, 1].")
    available_dark = (
        config.flash_period_us
        - config.fade_duration_us
        - config.peak_apply_time_us
    )
    if available_dark <= 3 * config.timing_edge_guard_us:
        raise ValueError("The configured dark interval is too short for timing validation.")


def build_fixed_flash_onsets(first_onset_us: int, config: TimingConfig) -> np.ndarray:
    """Generate the fixed R/G/B schedule from one detected first onset."""
    j = np.arange(len(config.flash_colours), dtype=np.int64)
    return (
        int(first_onset_us)
        + j * int(config.flash_period_us)
        + (j // 3) * int(config.cycle_extra_delay_us)
    )


def _count_between(
    prefix: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    polarity: int,
    bin_us: int,
) -> np.ndarray:
    lo = np.floor_divide(left, bin_us).astype(np.int64)
    hi = np.floor_divide(right, bin_us).astype(np.int64)
    return prefix[hi, polarity] - prefix[lo, polarity]


def evaluate_fixed_timing(
    counts: np.ndarray,
    first_times_us: np.ndarray | list[int],
    config: TimingConfig,
    prepared_prefix: np.ndarray | None = None,
) -> dict:
    """Score candidate first onsets against the configured fixed RGB sequence.

    The standardized scores are heuristic evidence values.  They are not
    hardware-synchronization confidence probabilities.
    """
    validate_timing_config(config)
    first_times = np.atleast_1d(first_times_us).astype(np.int64)
    bin_us = int(config.flash_detection_bin_us)
    prefix = (
        np.vstack((np.zeros((1, 2)), np.cumsum(counts, axis=0, dtype=float)))
        if prepared_prefix is None
        else prepared_prefix
    )
    duration_us = len(counts) * bin_us
    offsets = build_fixed_flash_onsets(0, config)
    quiet_us = max(
        int(config.first_red_baseline_us),
        3 * int(config.flash_period_us) + int(config.cycle_extra_delay_us),
    )

    if np.any(first_times < quiet_us) or np.any(
        first_times + offsets[-1] + config.flash_period_us > duration_us
    ):
        raise TimingDetectionError(
            "Insufficient pre-flash dark background or incomplete RGB sequence."
        )

    guard = int(config.timing_edge_guard_us)
    dark_start = int(config.fade_duration_us + config.peak_apply_time_us)
    late_start = dark_start + 2 * guard
    late_duration = int(config.flash_period_us - late_start)
    on_ratio = config.fade_duration_us / late_duration
    off_ratio = (2 * guard) / late_duration

    count_n = len(first_times)
    light_on_sum = np.zeros(count_n)
    dark_on_sum = np.zeros(count_n)
    edge_off_sum = np.zeros(count_n)
    dark_off_sum = np.zeros(count_n)
    rank = np.zeros(count_n)
    supported = np.zeros(count_n, dtype=int)
    early_supported = np.zeros(count_n, dtype=int)
    late_supported = np.zeros(count_n, dtype=int)
    first_supported = np.zeros(count_n, dtype=bool)
    per_flash: list[dict] = []

    for i, offset in enumerate(offsets):
        t = first_times + offset
        bright_on = _count_between(
            prefix, t, t + config.fade_duration_us, 1, bin_us
        )
        dark_on = _count_between(
            prefix,
            t + late_start,
            t + config.flash_period_us,
            1,
            bin_us,
        )
        edge_off = _count_between(
            prefix,
            t + dark_start - guard,
            t + dark_start + guard,
            0,
            bin_us,
        )
        dark_off = _count_between(
            prefix,
            t + late_start,
            t + config.flash_period_us,
            0,
            bin_us,
        )

        on_score = (
            bright_on - on_ratio * dark_on
        ) / np.sqrt(np.maximum(bright_on + on_ratio**2 * dark_on, 1.0))
        off_score = (
            edge_off - off_ratio * dark_off
        ) / np.sqrt(np.maximum(edge_off + off_ratio**2 * dark_off, 1.0))
        flash_supported = (
            (on_score >= config.timing_min_flash_score)
            | (off_score >= config.timing_min_flash_score)
        )

        light_on_sum += bright_on
        dark_on_sum += dark_on
        edge_off_sum += edge_off
        dark_off_sum += dark_off
        rank += 12 * np.tanh(on_score / 12) + 6 * np.tanh(off_score / 12)
        supported += flash_supported

        if i < 3:
            early_supported += flash_supported
        if i >= len(offsets) - 3:
            late_supported += flash_supported
        if i == 0:
            first_supported = flash_supported

        if count_n == 1:
            per_flash.append(
                {
                    "stage": f"{config.flash_colours[i]}{i // 3 + 1}",
                    "onset_after_read_us": int(t[0]),
                    "ramp_ON": float(bright_on[0]),
                    "late_dark_ON": float(dark_on[0]),
                    "end_edge_OFF": float(edge_off[0]),
                    "late_dark_OFF": float(dark_off[0]),
                    "ON_contrast_score": float(on_score[0]),
                    "OFF_edge_score": float(off_score[0]),
                    "supported": bool(flash_supported[0]),
                }
            )

    sequence_on = (
        light_on_sum - on_ratio * dark_on_sum
    ) / np.sqrt(np.maximum(light_on_sum + on_ratio**2 * dark_on_sum, 1.0))
    sequence_off = (
        edge_off_sum - off_ratio * dark_off_sum
    ) / np.sqrt(np.maximum(edge_off_sum + off_ratio**2 * dark_off_sum, 1.0))

    before_on = _count_between(
        prefix, first_times - quiet_us, first_times, 1, bin_us
    )
    quiet_ratio = quiet_us / (len(offsets) * late_duration)
    startup_score = (
        before_on - quiet_ratio * dark_on_sum
    ) / np.sqrt(np.maximum(before_on + quiet_ratio**2 * dark_on_sum, 1.0))

    cycle_us = min(
        3 * int(config.flash_period_us) + int(config.cycle_extra_delay_us),
        int(offsets[-1]) + int(config.flash_period_us),
    )
    after_start_on = _count_between(
        prefix, first_times, first_times + cycle_us, 1, bin_us
    )
    start_ratio = cycle_us / quiet_us
    start_transition = (
        after_start_on - start_ratio * before_on
    ) / np.sqrt(np.maximum(after_start_on + start_ratio**2 * before_on, 1.0))

    required_supported = max(
        1,
        int(math.ceil(len(offsets) * config.timing_min_supported_fraction)),
    )
    cycle_required = max(
        1,
        int(math.ceil(min(3, len(offsets)) * config.timing_min_supported_fraction)),
    )

    passed = (
        first_supported
        & (supported >= required_supported)
        & (early_supported >= cycle_required)
        & (late_supported >= cycle_required)
        & (startup_score <= config.timing_max_startup_score)
        & (start_transition >= config.timing_min_sequence_score)
        & (np.maximum(sequence_on, sequence_off) >= config.timing_min_sequence_score)
    )

    return {
        "passed": passed,
        "rank": rank / len(offsets),
        "supported": supported,
        "required_supported": required_supported,
        "sequence_ON_score": sequence_on,
        "sequence_OFF_score": sequence_off,
        "startup_score": startup_score,
        "startup_transition_score": start_transition,
        "quiet_before_us": quiet_us,
        "per_flash": per_flash,
    }


def timing_summary(evaluation: dict, index: int = 0) -> dict:
    return {
        "automatic_timing_passed": bool(evaluation["passed"][index]),
        "supported_flashes": int(evaluation["supported"][index]),
        "required_supported_flashes": int(evaluation["required_supported"]),
        "rank_score": float(evaluation["rank"][index]),
        "sequence_ON_score": float(evaluation["sequence_ON_score"][index]),
        "sequence_OFF_score": float(evaluation["sequence_OFF_score"][index]),
        "startup_activity_score": float(evaluation["startup_score"][index]),
        "startup_transition_score": float(
            evaluation["startup_transition_score"][index]
        ),
        "quiet_before_us": int(evaluation["quiet_before_us"]),
    }


def detect_first_flash_automatically(
    counts: np.ndarray,
    config: TimingConfig,
    available_end_us: int | None = None,
    candidate_min_us: int = 0,
    candidate_max_us: int | None = None,
) -> tuple[int, dict, list[dict]]:
    """Find one first activity transition supported by the full fixed RGB schedule."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 2 or not len(counts):
        raise TimingDetectionError("No ON/OFF timing bins are available.")

    bin_us = int(config.flash_detection_bin_us)
    available = (
        len(counts) * bin_us
        if available_end_us is None
        else min(int(available_end_us), len(counts) * bin_us)
    )
    quiet = max(
        int(config.first_red_baseline_us),
        3 * int(config.flash_period_us) + int(config.cycle_extra_delay_us),
    )
    last_offset = int(build_fixed_flash_onsets(0, config)[-1])
    first_max = available - last_offset - int(config.flash_period_us)
    if candidate_max_us is not None:
        first_max = min(first_max, int(candidate_max_us))
    first_min = int(math.ceil(max(quiet, candidate_min_us) / bin_us)) * bin_us

    if first_max < first_min:
        raise TimingDetectionError(
            "The data cannot cover both the pre-flash baseline and the complete RGB sequence."
        )

    candidates = np.arange(first_min, first_max + 1, bin_us, dtype=np.int64)
    prefix = np.vstack((np.zeros((1, 2)), np.cumsum(counts, axis=0, dtype=float)))

    parts = []
    for begin in range(0, len(candidates), 50_000):
        parts.append(
            evaluate_fixed_timing(
                counts,
                candidates[begin: begin + 50_000],
                config,
                prepared_prefix=prefix,
            )
        )

    vector_keys = (
        "passed",
        "rank",
        "supported",
        "sequence_ON_score",
        "sequence_OFF_score",
        "startup_score",
        "startup_transition_score",
    )
    evaluated = {
        key: np.concatenate([part[key] for part in parts]) for key in vector_keys
    }
    evaluated.update(
        {
            "required_supported": parts[0]["required_supported"],
            "quiet_before_us": quiet,
        }
    )

    good = np.flatnonzero(evaluated["passed"])
    if not len(good):
        best = int(np.argmax(evaluated["rank"]))
        diagnostics = timing_summary(evaluated, best)
        diagnostics["rejected_best_candidate_after_read_us"] = int(candidates[best])
        raise TimingDetectionError(
            "No candidate satisfies both the startup baseline and the fixed RGB timing pattern.",
            diagnostics,
        )

    neighbourhood_us = max(
        bin_us,
        min(int(config.fade_duration_us), int(config.flash_period_us) // 2),
    )
    spread_limit = max(
        3 * bin_us,
        min(int(config.fade_duration_us), int(config.dark_interval_us)) // 2,
    )
    cuts = np.flatnonzero(np.diff(candidates[good]) > quiet) + 1
    failures: list[dict] = []

    for group in np.split(good, cuts):
        local = group[candidates[group] <= candidates[group[0]] + neighbourhood_us]
        best_score = float(np.max(evaluated["rank"][local]))
        ties = local[
            np.isclose(evaluated["rank"][local], best_score, rtol=0, atol=1e-9)
        ]
        selected = int(ties[len(ties) // 2])
        near = group[
            evaluated["rank"][group]
            >= best_score - max(0.1, abs(best_score) * 0.02)
        ]
        spread = int(candidates[near[-1]] - candidates[near[0]] + bin_us)
        diagnostics = timing_summary(evaluated, selected)
        diagnostics.update(
            {
                "source": "automatic_first_activity_with_fixed_RGB_check",
                "first_onset_before_sync_us": int(candidates[selected]),
                "near_best_start_range_us": [
                    int(candidates[near[0]]),
                    int(candidates[near[-1]]),
                ],
                "phase_score_width_us": spread,
                "phase_score_width_limit_us": spread_limit,
                "bin_us": bin_us,
                "note": (
                    "Heuristic timing evidence only. RGB identity assumes the "
                    "recording starts before the first red stage."
                ),
            }
        )
        if spread <= spread_limit:
            break
        diagnostics["automatic_timing_passed"] = False
        failures.append(diagnostics)
    else:
        diagnostics = dict(failures[0])
        diagnostics["ambiguous_activity_groups_in_block"] = len(failures)
        raise TimingDetectionError(
            "Candidate onset phase is ambiguous in this activity block.", diagnostics
        )

    diagnostics["earlier_ambiguous_activity_groups_in_block"] = len(failures)
    first = int(candidates[selected]) + int(config.sync_offset_us)
    checked = evaluate_fixed_timing(counts, [first], config)
    diagnostics.update(timing_summary(checked))
    diagnostics["first_onset_after_sync_us"] = first

    if not diagnostics["automatic_timing_passed"]:
        raise TimingDetectionError(
            "The global sync offset makes the fixed timing check fail.", diagnostics
        )

    return first, diagnostics, checked["per_flash"]
