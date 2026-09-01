# Event-Based Colour Response Analysis and Pseudo-Colour Reconstruction

This repository documents my 2026 summer research project at the **University of Manchester**, investigating colour-related event responses and pseudo-colour reconstruction using a monochrome event camera under sequential RGB illumination.

The project explores whether relative colour information can be extracted from asynchronous event measurements by analysing the sensor response to controlled **red, green, and blue illumination**.

The work covers:

- event-stream decoding,
- temporal synchronisation,
- ROI-based event-response analysis,
- relative RGB response estimation,
- illumination-duration robustness analysis,
- pseudo-colour event visualisation.

The current system should be regarded as an **experimental colour-response analysis framework**, rather than a calibrated or universally robust RGB reconstruction system.

---

## Project Demo

<p align="center">
  <img src="results/demos/pseudo_colour_result.gif" width="850">
</p>

The animation above shows a representative pseudo-colour event output obtained under the **1 ms illumination + 500 µs dark-interval condition**.

Spatial and temporal information is derived from active event pixels, while colour is assigned according to the relative RGB response measured within selected regions of interest.

Additional experiments using different illumination durations showed that the estimated colour-response vector is sensitive to the temporal illumination profile.

The visualisation should therefore be interpreted as a **condition-specific ROI-level pseudo-colour representation**, rather than a fully calibrated or robust RGB reconstruction.

---

## Overview

Conventional frame-based cameras capture complete images at fixed time intervals.

In contrast, an **event camera** records changes in logarithmic brightness asynchronously. Each pixel independently generates an event when the local brightness change exceeds a predefined contrast threshold.

An event can be represented as:

```text
e = (x, y, t, p)
```

where:

- `x, y` represent the pixel coordinates,
- `t` represents the timestamp,
- `p` represents the event polarity.

This sensing principle provides several advantages:

- high temporal resolution,
- low latency,
- high dynamic range,
- reduced motion blur,
- sparse data representation.

However, a monochrome event camera does not directly provide conventional RGB colour information.

This project investigates whether relative colour-related information can be inferred by illuminating the scene sequentially with **red, green, and blue light** and analysing the corresponding event responses.

The overall processing pipeline is:

```text
Sequential RGB Illumination
          ↓
Monochrome Event Camera
          ↓
Event Stream Recording
          ↓
DAT Event Decoding
          ↓
Flash / Phase Detection
          ↓
Temporal Synchronisation
          ↓
ROI Event Response Extraction
          ↓
Relative RGB Response Estimation
          ↓
Pseudo-Colour Mapping
```

---

## Research Questions

This project investigates the following questions:

1. How does a monochrome event camera respond to sequential red, green, and blue illumination?

2. Can event density measured under different illumination colours be used to estimate a relative RGB response?

3. How can the RGB illumination sequence be synchronised with an asynchronous event stream?

4. How do ON and OFF events behave during controlled illumination changes?

5. How sensitive is the estimated RGB response to illumination duration and temporal integration conditions?

6. What factors limit reliable colour-response estimation using event-camera measurements alone?

7. How could this approach be extended toward multimodal perception using event cameras, conventional RGB cameras, LiDAR, and other sensing modalities?

---

## Experimental Setup

The experimental system consists of:

- **Event Camera:** Prophesee EVK4 monochrome event camera
- **Resolution:** 1280 × 720
- **Illumination:** Sequential red, green, and blue light
- **Illumination Control:** Arduino-based PWM control
- **Event Representation:** `(x, y, t, p)`
- **Main Processing Language:** Python
- **Core Libraries:** NumPy, OpenCV, Matplotlib, Pillow

<p align="center">
  <img src="results/figures/experimental_setup.jpg" width="750">
</p>

The RGB illumination sequence is repeated over multiple cycles.

During each illumination stage, the asynchronous response of the event camera is recorded.

Selected regions of interest are then analysed to compare their relative responses under red, green, and blue illumination.

---

# Methodology

## 1. Event Stream Decoding

The event-camera recordings are stored as Prophesee-style DAT files.

A Python processing pipeline was developed to decode the asynchronous event stream and extract:

- pixel coordinates,
- timestamps,
- event polarity.

The basic event representation is:

```text
Event = (x, y, timestamp, polarity)
```

Large recordings are processed in chunks to reduce memory requirements and enable efficient analysis of long event sequences.

The decoded events form the basis for all subsequent temporal and spatial processing.

---

## 2. Temporal Event Accumulation

Although an event camera operates asynchronously, short temporal windows can be used to accumulate events for analysis and visualisation.

For a selected interval:

```text
[t_start, t_end]
```

events within the interval are extracted and accumulated according to their spatial coordinates and polarity.

This provides a temporary image-like representation while retaining the original asynchronous event stream for quantitative analysis.

---

## 3. Flash Detection

One of the main challenges in the experiment is identifying when the red, green, and blue illumination stages occur within the event stream.

An initial approach attempted to locate each illumination stage independently by selecting the time windows containing the largest number of events.

However, this method was found to be unreliable.

PWM-controlled illumination can generate several strong event bursts within a single physical illumination stage.

Therefore:

```text
Maximum event count ≠ physical illumination onset
```

The strongest event peak does not necessarily correspond to the actual beginning of an illumination stage.

This observation motivated the development of a fixed-sequence temporal synchronisation strategy.

---

## 4. Temporal Synchronisation

Accurate temporal alignment is critical because the measured colour response depends directly on which events are assigned to each red, green, and blue illumination stage.

Rather than independently locating the strongest event peak for every stage, the current pipeline uses a two-step strategy:

1. detect a plausible onset for the first illumination stage;
2. generate all subsequent RGB response windows from a fixed illumination timing model.

<p align="center">
  <img src="results/figures/flash_detection.png" width="850">
</p>

The first onset is identified by searching for a transition from relatively low event activity to an event pattern that is statistically consistent with the expected RGB sequence.

Once this initial phase is selected, the remaining illumination stages are calculated using:

```text
t_j = t_0 + j × T
```

where:

- `t_0` is the detected initial illumination phase,
- `j` is the illumination-stage index,
- `T` is the configured adjacent-stage period.

For the representative **1 ms illumination** experiment shown in this repository:

```text
Illumination / fade duration: 1000 µs
Dark interval:               500 µs
Adjacent-stage period:       1500 µs
```

The later RGB stages are **not independently shifted toward nearby event peaks**.

This avoids fitting each measurement window directly to potentially misleading PWM-induced local maxima.

### Timing Interpretation

A successful automatic timing check indicates that the observed event activity is statistically consistent with the configured RGB timing model.

It does **not** imply exact hardware-level synchronisation between the illumination controller and the event camera.

In particular, the event-derived onset is not necessarily identical to the physical instant at which the LED controller begins its illumination ramp.

This distinction becomes increasingly important when comparing different illumination durations.

### Key Observation

> **The pipeline detects one initial illumination phase and preserves a fixed RGB timing model, rather than independently fitting every flash to an event peak.**

---

## 5. Region-of-Interest Analysis

Polygonal **regions of interest (ROIs)** are selected from the recorded scene.

For each ROI, event activity is measured during the corresponding red, green, and blue illumination stages.

The analysis includes:

- ON-event count,
- OFF-event count,
- total event count,
- ROI area,
- area-normalised event density.

<p align="center">
  <img src="results/figures/roi_response.png" width="850">
</p>

Because different ROIs may contain different numbers of pixels, raw event counts cannot always be compared directly.

An area-normalised response is therefore calculated as:

```text
Event Density = Number of Events / ROI Area
```

For nested regions, an inner ROI can optionally be excluded from the statistical mask of a larger containing ROI. This allows a local test or defect region to be analysed separately from its surrounding reference surface.

---

## 6. RGB Response Estimation

For each ROI, the event responses measured under red, green, and blue illumination are used to construct a **relative RGB response vector**.

The current baseline reconstruction method uses **ON-event density** as the colour-response feature.

Because the RGB illumination sequence is repeated multiple times, the ON-event density is measured independently for each illumination colour and repetition.

For three repetitions:

```text
Red:   R1, R2, R3
Green: G1, G2, G3
Blue:  B1, B2, B3
```

For each illumination colour, the maximum ON-event density across the repeated measurements is selected:

```text
R_peak = max(R1, R2, R3)
G_peak = max(G1, G2, G3)
B_peak = max(B1, B2, B3)
```

The unnormalised response is:

```text
RGB_peak = [R_peak, G_peak, B_peak]
```

The vector is then sum-normalised:

```text
RGB_response =
[R_peak, G_peak, B_peak]
/
(R_peak + G_peak + B_peak)
```

Therefore, for a valid non-zero response:

```text
R + G + B = 1
```

<p align="center">
  <img src="results/figures/rgb_response.png" width="850">
</p>

The resulting vector represents the **relative ON-event response of the complete sensor–illumination–surface system** under the three illumination channels.

For example:

```text
RGB_response = [0.52, 0.32, 0.16]
```

means that approximately 52% of the measured three-channel ON-peak response is associated with red illumination, 32% with green illumination, and 16% with blue illumination.

This should **not** be interpreted as a conventional RGB pixel value such as:

```text
RGB = (0.52, 0.32, 0.16)
```

because the measured event response depends on the combined effects of:

- illumination intensity,
- illumination spectrum,
- surface reflectance,
- sensor spectral sensitivity,
- event contrast thresholds,
- temporal illumination behaviour,
- previous illumination state,
- motion,
- noise.

The recovered vector is therefore better interpreted as a:

> **relative colour-related event-response signature under controlled RGB illumination**

rather than as calibrated surface colour.

### Display Colour Scaling

The quantitative response vector is preserved for analysis.

For visualisation only, it is additionally scaled by its largest component:

```text
RGB_display =
RGB_response / max(RGB_response)
```

This maximum-based scaling is used only to generate a visible pseudo-colour representation.

It does **not** modify the quantitative response vector.

Additional display-only brightness and desaturation adjustments may also be applied when generating the final pseudo-colour output.

---

## 7. ON and OFF Event Analysis

Both ON and OFF events were investigated during the experiments.

An ON event is generally produced when logarithmic brightness increases sufficiently, while an OFF event is generated when logarithmic brightness decreases sufficiently.

The measured response depends on several factors:

- the previous illumination state,
- the direction of the illumination transition,
- PWM timing,
- surface reflectance,
- local contrast,
- sensor thresholds,
- motion,
- noise.

Therefore:

```text
ON/OFF Event Count ≠ Absolute Optical Intensity
```

The current method consequently focuses on relative event responses under controlled illumination rather than directly interpreting event counts as RGB intensity values.

---

## 8. Illumination-Duration Robustness Analysis

To test whether the estimated response remained stable under different temporal illumination conditions, additional recordings were analysed using four illumination durations:

```text
500 µs
1 ms
10 ms
100 ms
```

with a constant:

```text
500 µs dark interval
```

The comparison was performed using the same analysis framework and shared ROI definitions where possible.

In addition to the original `ON_PEAK` estimator, alternative aggregation strategies such as median-based ON responses and positive net ON–OFF responses were investigated.

The experiments showed that the estimated RGB response is **not invariant to illumination duration**.

This indicates that the recovered response depends not only on the surface itself, but also on the temporal illumination conditions and event-generation dynamics.

The duration-analysis script is provided as:

```text
run_duration_experiment.py
```

The purpose of this analysis is not to select parameters that visually reproduce an expected colour, but to evaluate the stability and limitations of the event-derived response.

---

## 9. Pseudo-Colour Reconstruction

After estimating the relative RGB response of each ROI, the response is mapped back onto active event pixels.

The resulting representation combines:

```text
Event-camera spatial information
            +
Event-camera temporal information
            +
ROI-level RGB response estimate
```

to generate a pseudo-colour event output.

Spatial information continues to come directly from active event pixels.

Colour information is assigned according to the relative RGB response measured for the corresponding ROI.

The current output should therefore be interpreted as:

> **ROI-level pseudo-colour reconstruction based on relative event responses under controlled sequential RGB illumination.**

It is not equivalent to conventional RGB video reconstruction.

---

# Key Findings

## 1. Maximum Event Count Is Not a Reliable Illumination Trigger

PWM-controlled illumination can produce multiple strong event bursts within one RGB illumination stage.

Therefore, independently searching for the largest event-count peak can lead to incorrect temporal interpretation.

This makes illumination timing an important component of the processing pipeline.

---

## 2. Temporal Synchronisation Is Critical

Event cameras respond strongly to rapid brightness transitions.

Even relatively small differences in the temporal measurement window can substantially change the number and distribution of recorded events.

The current automatic timing method provides a useful consistency check, but it should not be interpreted as exact hardware-level synchronisation.

---

## 3. Colour Response Is Sensitive to Illumination Duration

Experiments using **500 µs, 1 ms, 10 ms, and 100 ms illumination durations**, while maintaining a **500 µs dark interval**, produced different relative RGB response vectors.

This shows that the current event-derived colour signature is sensitive to the temporal illumination profile.

A useful conceptual description is:

```text
Measured colour-related response
        =
Surface response
        +
Illumination spectrum and intensity
        +
Sensor spectral / threshold response
        +
Temporal illumination dynamics
```

The method should therefore be regarded as a **condition-dependent colour-response analysis framework**, rather than a robust colour-reconstruction method across arbitrary illumination timings.

---

## 4. Different Surfaces Produce Different RGB Responses

Selected surfaces generate different relative responses under red, green, and blue illumination.

The measured response is influenced by the interaction between:

```text
Illumination Spectrum
        ×
Surface Reflectance
        ×
Sensor Spectral Response
```

This indicates that colour-related information can be present in the event response even when the sensor itself is monochrome.

---

## 5. Event Count Is Not Equivalent to RGB Intensity

An event camera measures changes in logarithmic brightness rather than absolute optical intensity.

Therefore:

```text
Event Count ≠ Absolute RGB Intensity
```

The measured response is affected by:

- sensor contrast threshold,
- illumination intensity,
- illumination timing,
- previous illumination state,
- surface reflectance,
- motion,
- local scene structure,
- noise.

Direct conversion from event count to physically accurate RGB intensity therefore requires additional modelling and calibration.

---

## 6. Relative Colour Information Can Still Be Extracted

Although calibrated RGB reconstruction cannot be obtained directly from event counts alone, controlled sequential RGB illumination produces distinguishable colour-related event responses.

Under selected experimental conditions, these responses can be used to generate an ROI-level pseudo-colour representation of the event stream.

The robustness experiments, however, demonstrate that this representation remains dependent on the temporal illumination configuration.

---

# Limitations

## Illumination Calibration

The red, green, and blue light sources may not produce identical optical power.

Differences in event responses may therefore be caused partly by LED intensity rather than surface reflectance alone.

---

## Sensor Spectral Sensitivity

The monochrome event sensor may have different sensitivities at different wavelengths.

Quantitative colour reconstruction would require a calibrated model of the sensor's spectral response.

---

## Event Thresholds

Events are generated only when the change in logarithmic brightness exceeds the sensor's internal contrast threshold.

Consequently, event density is not linearly proportional to absolute optical intensity.

---

## PWM Modulation

PWM-driven illumination introduces additional brightness transitions.

These transitions generate additional ON and OFF events and can influence both temporal alignment and measured response magnitude.

---

## Illumination-Duration Sensitivity

The estimated RGB response was found to vary with the duration of the RGB illumination ramp.

Experiments using 500 µs, 1 ms, 10 ms, and 100 ms illumination durations produced different relative response vectors even though the dark interval remained fixed.

This suggests that the current response is sensitive to:

- illumination ramp speed,
- temporal integration conditions,
- PWM behaviour,
- event-camera contrast thresholds,
- the relationship between event-derived onset and physical illumination onset.

Further hardware synchronisation and radiometric calibration would be required before the method could be considered robust across different temporal illumination conditions.

---

## Timing Reference

The automatically detected event onset represents a statistically supported transition in the event stream.

It is not guaranteed to correspond exactly to the physical start time of the LED illumination ramp.

A hardware trigger or independent optical timing reference would provide more precise synchronisation.

---

## Motion

Object motion or camera motion also generates events.

In a dynamic environment, the measured event stream therefore contains a mixture of:

```text
Illumination-induced events
            +
Motion-induced events
```

Separating these two sources remains an important challenge.

---

## ROI-Level Estimation

The current implementation primarily estimates colour responses at the ROI level.

A more advanced system could investigate:

- local adaptive estimation,
- dense spatial estimation,
- pixel-level spectral response modelling,
- learning-based reconstruction.

---

## Response Estimator

The current baseline method uses the maximum ON-event response across repeated RGB cycles.

While simple and interpretable, this `ON_PEAK` estimator can be sensitive to unusually strong repetitions.

Median-based and ON–OFF-based alternatives were therefore also explored during robustness analysis.

---

# Future Work

## 1. Hardware Synchronisation

A direct synchronisation signal between the illumination controller and the event-camera recording system could provide precise physical illumination timestamps.

This would reduce uncertainty between the actual LED onset and the onset inferred from event activity.

---

## 2. Illumination Calibration

Future experiments could measure the optical output of each RGB illumination channel.

This would help distinguish differences caused by:

- LED intensity,
- surface reflectance,
- sensor sensitivity.

---

## 3. Sensor Spectral Calibration

The wavelength-dependent response of the event sensor could be characterised experimentally.

A calibrated spectral sensitivity model could then be incorporated into colour-response estimation.

---

## 4. RGB Camera Ground Truth

A conventional RGB camera could be added to provide reference colour measurements.

The system could then compare:

```text
Event-derived colour response
            ↓
RGB camera ground truth
```

This would enable quantitative evaluation using metrics such as:

- RGB error,
- chromaticity error,
- colour distance,
- reconstruction consistency,
- colour classification accuracy.

---

## 5. Robust Temporal Response Modelling

Future work could investigate how colour-related event responses vary across normalised stages of the illumination ramp rather than using only fixed absolute time windows.

Alternative estimators could also include:

- median response,
- trimmed mean,
- uncertainty-weighted response,
- calibrated ON/OFF response models,
- learning-based mappings.

---

## 6. Event + RGB Sensor Fusion

A longer-term extension is to combine the complementary properties of event cameras and conventional RGB cameras.

An event camera provides:

- high temporal resolution,
- low latency,
- reduced motion blur,
- high dynamic range.

An RGB camera provides:

- absolute intensity information,
- conventional colour information,
- dense spatial appearance.

Combining these modalities could support more robust perception in challenging and highly dynamic environments.

---

## 7. Event + RGB + LiDAR Perception

The project could also be extended toward multimodal robotic perception by combining:

```text
Event Camera
     +
RGB Camera
     +
LiDAR
```

Such a system could exploit:

- temporal information from event cameras,
- appearance and colour information from RGB cameras,
- depth and geometric information from LiDAR.

This direction is particularly relevant to robotic perception and autonomous systems.

---

## 8. Motion Compensation

Future work could investigate:

- event-based optical flow,
- feature tracking,
- motion estimation,
- object tracking,

to separate motion-generated events from illumination-generated events.

---

# Repository Structure

```text
event-camera-colour-reconstruction/
│
├── README.md
├── .gitignore
├── requirements.txt
├── run_pipeline.py
├── run_duration_experiment.py
│
├── src/
│   ├── dat_decoder.py
│   ├── flash_detection.py
│   ├── temporal_alignment.py
│   ├── roi_analysis.py
│   └── colour_mapping.py
│
├── results/
│   ├── figures/
│   │   ├── experimental_setup.jpg
│   │   ├── flash_detection.png
│   │   ├── roi_response.png
│   │   └── rgb_response.png
│   │
│   └── demos/
│       └── pseudo_colour_result.gif
│
└── docs/
    └── technical_report.pdf
```

---

# Software

The project primarily uses:

- **Python**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Pillow**

Project dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
python -m pip install -r requirements.txt
```

---

# Running the Main Pipeline

The main single-recording analysis can be run using:

```bash
python run_pipeline.py
```

The input DAT recording and experimental timing parameters should first be configured inside `run_pipeline.py`.

The main pipeline performs:

```text
DAT decoding
    ↓
automatic first-phase detection
    ↓
fixed RGB temporal alignment
    ↓
ROI selection
    ↓
event-response measurement
    ↓
relative RGB estimation
    ↓
pseudo-colour visualisation
```

---

# Running the Illumination-Duration Experiment

The cross-duration robustness analysis can be run using:

```bash
python run_duration_experiment.py
```

The script is designed to compare the four experimental conditions:

```text
r1 → 100 ms illumination + 500 µs dark
r2 → 10 ms illumination  + 500 µs dark
r3 → 1 ms illumination   + 500 µs dark
r4 → 500 µs illumination + 500 µs dark
```

It evaluates:

- RGB response variation across illumination durations,
- repeatability across RGB cycles,
- time-normalised event-response rates,
- alternative response estimators,
- cross-duration response-vector differences.

Generated quantitative experiment directories are excluded from version control by `.gitignore`.

---

# Data Availability

The original event-camera DAT recordings are not included in this public repository because of their large file size and data-sharing considerations.

The public repository contains the processing code, representative results, documentation, and analysis framework.

No confidential, proprietary, or internally restricted research data are intended to be included.

---

# Technical Report

A detailed technical report covering the experimental design, processing methodology, configuration, limitations, and reproducibility of the project is available here:

[Technical Report](docs/technical_report.pdf)

---

# Project Context

This work was conducted during a **2026 summer research project at the University of Manchester**.

The project forms part of my broader research interests in:

- Event-Based Vision
- Computer Vision
- Robotic Perception
- Multimodal Sensor Fusion
- LiDAR Perception
- Low-Latency Visual Sensing
- Autonomous Systems

---

# Author

**Shize Yang**

BEng (Hons) Electronic Engineering  
University of Manchester

Research interests:

**Event-Based Vision | Computer Vision | Robotic Perception | Sensor Fusion | Autonomous Systems**

GitHub: [Bestzivvv](https://github.com/Bestzivvv)

LinkedIn: [Ziv Yang](https://www.linkedin.com/in/ziv-yang-8b0aa5349)

---

# Disclaimer

This repository documents an undergraduate research project and ongoing experimental work.

The reported RGB response represents **relative event behaviour under controlled sequential RGB illumination**.

The pseudo-colour output is intended for visualisation and should not be interpreted as calibrated surface RGB or conventional RGB video reconstruction.

Cross-duration experiments further indicate that the estimated colour response is sensitive to the temporal illumination configuration. The current implementation should therefore be regarded as an experimental analysis framework rather than a universally robust colour-reconstruction system.
