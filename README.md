# Event-Based Colour Response Analysis and Pseudo-Colour Reconstruction

This repository documents my 2026 summer research project at the **University of Manchester**, focusing on colour-response analysis and pseudo-colour reconstruction using a monochrome event camera under sequential RGB illumination.

The project investigates whether relative colour information can be extracted from asynchronous event measurements by analysing the sensor response to controlled **red, green, and blue illumination**.

The work covers event-stream decoding, temporal synchronisation, ROI-based event-response analysis, RGB response estimation, and pseudo-colour visualisation.

---

## Project Demo

<p align="center">
  <img src="results/demos/pseudo_colour_result.gif" width="850">
</p>

The animation above shows an example of the final pseudo-colour event output.

Spatial and temporal information is derived from active event pixels, while colour is assigned according to the relative RGB response measured within selected regions of interest.

The current output should be interpreted as an **ROI-level pseudo-colour representation**, rather than a fully calibrated RGB reconstruction.

---

## Overview

Conventional frame-based cameras capture complete images at fixed time intervals.

In contrast, an **event camera** records changes in logarithmic brightness asynchronously. Each pixel independently generates an event when the local brightness change exceeds a predefined threshold.

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

However, a monochrome event camera does not directly provide RGB colour information.

This project explores whether relative colour information can be inferred by illuminating the scene sequentially with **red, green, and blue light** and analysing the corresponding event responses.

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
RGB Response Vector Estimation
          ↓
Pseudo-Colour Mapping
```

The current system should therefore be regarded as an experimental framework for analysing colour-related event responses rather than a fully calibrated RGB reconstruction system.

---

## Research Questions

This project investigates the following questions:

1. How does a monochrome event camera respond to sequential red, green, and blue illumination?

2. Can event density measured under different illumination colours be used to estimate a relative RGB response?

3. How can the RGB illumination sequence be reliably synchronised with an asynchronous event stream?

4. How do ON and OFF events behave during controlled illumination changes?

5. What factors limit reliable colour reconstruction using event-camera measurements alone?

6. How could this approach be extended toward multimodal perception using event cameras, conventional RGB cameras, and other sensing modalities?

---

## Experimental Setup

The experimental system consists of:

- **Event Camera:** Prophesee EVK4 monochrome event camera
- **Resolution:** 1280 × 720
- **Illumination:** Sequential red, green, and blue light
- **Illumination Control:** Arduino-based control
- **Event Representation:** `(x, y, t, p)`
- **Main Processing Language:** Python
- **Core Libraries:** NumPy and OpenCV

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

This makes it possible to create temporary event representations while retaining the original asynchronous event stream for quantitative analysis.

---

## 3. Flash Detection

One of the main challenges in the experiment is identifying when the red, green, and blue illumination stages occur within the event stream.

An initial approach attempted to locate each illumination stage independently by selecting the time windows containing the largest number of events.

However, this method was found to be unreliable.

PWM-controlled illumination can generate several strong event bursts within a single illumination stage.

Therefore:

```text
Maximum event count ≠ illumination onset
```

The strongest event peak does not necessarily correspond to the actual beginning of the physical illumination stage.

This observation motivated the development of a more robust temporal synchronisation strategy.

---

## 4. Temporal Synchronisation

Accurate temporal alignment is critical because the colour-response measurements depend directly on which events are assigned to each red, green, and blue illumination stage.

Rather than independently locating the strongest event peak for every illumination stage, the current pipeline uses a two-step strategy:

1. detect a plausible onset for the first illumination stage;
2. generate all subsequent RGB response windows from a fixed illumination timing model.

<p align="center">
  <img src="results/figures/flash_detection.png" width="850">
</p>

The first onset is identified by searching for a transition from relatively low event activity to an event pattern that is statistically consistent with the expected RGB illumination sequence.

Once this initial onset is selected, the remaining illumination stages are calculated using:

```text
t_j = t_0 + j × T
```

where:

- `t_0` is the detected first illumination onset,
- `j` is the illumination-stage index,
- `T` is the configured adjacent-stage period.

In the current software configuration, the nominal timing is:

```text
Fade duration: 1000 µs
Dark interval: 500 µs
Adjacent-stage period: 1500 µs
```

The later RGB stages are therefore **not independently shifted toward nearby event peaks**.

This is important because PWM-controlled illumination can generate multiple strong event bursts within a single physical illumination stage. As a result:

```text
Maximum event count ≠ physical illumination onset
```

Independent peak matching could therefore produce visually convincing but physically inconsistent alignment.

### Timing Interpretation

A successful automatic timing check indicates that the observed event activity is statistically consistent with the configured fixed RGB timing model.

It does **not** imply exact hardware-level synchronisation between the illumination controller and the event camera.

In particular, differences between the configured period and the true physical illumination timing may produce accumulated phase error across repeated RGB cycles.

For this reason, the timing diagnostic should be inspected when analysing a new recording, and an experimentally measured illumination period should be used whenever available.

### Key Observation

> **The pipeline detects one initial illumination phase and preserves a fixed RGB timing model, rather than independently fitting every flash to an event peak.**

This design keeps the temporal analysis tied to the experimental timing assumptions and reduces the risk of overfitting the event stream.

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

This enables more meaningful comparison between different surfaces or regions.

---

## 6. RGB Response Estimation

For each ROI, the event responses measured under red, green, and blue illumination are used to construct a **relative RGB response vector**.

The current reconstruction method uses **ON-event density** as the colour-response feature.

Because the RGB illumination sequence is repeated multiple times, the ON-event density is first measured independently for each illumination colour and each repetition.

For three repetitions, the measurements can be written as:

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

The unnormalised colour-response vector is therefore:

```text
RGB_peak = [R_peak, G_peak, B_peak]
```

The vector is then normalised by the sum of its three components:

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

The resulting vector represents the **relative ON-event response of the event-camera system** under the three illumination channels.

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
- motion,
- noise.

The recovered vector is therefore better interpreted as a:

> **relative colour-related event-response signature under controlled RGB illumination**

rather than as calibrated surface colour.

### Display Colour Scaling

The quantitative RGB response vector described above is preserved for analysis and exported results.

For visualisation only, the response vector is additionally scaled by its maximum component:

```text
RGB_display =
RGB_response / max(RGB_response)
```

This maximum-based scaling is used only to generate a visible pseudo-colour representation.

It does **not** modify the quantitative RGB response vector used in the analysis.

Additional display-only brightness and desaturation adjustments may also be applied when generating the final pseudo-colour event output.

These visual adjustments are kept separate from the measured response values.

---

## 7. ON and OFF Event Analysis

Both ON and OFF events were investigated during the experiments.

An ON event is generally produced when the logarithmic brightness observed by a pixel increases sufficiently.

An OFF event is generally produced when the logarithmic brightness decreases sufficiently.

However, the measured ON/OFF response depends on several factors:

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

## 8. Pseudo-Colour Reconstruction

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

<p align="center">
  <img src="results/demos/pseudo_colour_result.gif" width="850">
</p>

Spatial information continues to come directly from active event pixels.

Colour information is assigned according to the relative RGB response measured for the corresponding ROI.

The current output should therefore be interpreted as:

> **ROI-level pseudo-colour reconstruction based on relative event responses under sequential RGB illumination.**

It is not equivalent to conventional RGB video reconstruction.

---

# Key Findings

## 1. Maximum Event Count Is Not a Reliable Illumination Trigger

PWM-controlled illumination can produce multiple strong event bursts within one RGB illumination stage.

Therefore, independently searching for the largest event-count peak can lead to incorrect temporal alignment.

This makes illumination timing one of the most important components of the processing pipeline.

---

## 2. Temporal Synchronisation Is Critical

Event cameras respond strongly to rapid brightness transitions.

Even relatively small timing errors can significantly alter the number of events measured inside a short response window.

Reliable synchronisation between the illumination system and the event stream is therefore essential for consistent RGB-response analysis.

---

## 3. Different Surfaces Produce Different RGB Responses

Selected surfaces generate different relative responses under red, green, and blue illumination.

The measured response is influenced by the interaction between:

```text
Illumination Spectrum
        ×
Surface Reflectance
        ×
Sensor Spectral Response
```

This suggests that colour-related information can be present in the event response even when the sensor itself is monochrome.

---

## 4. Event Count Is Not Equivalent to RGB Intensity

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

For this reason, direct conversion from event count to physically accurate RGB intensity is not possible without additional modelling and calibration.

---

## 5. Relative Colour Information Can Still Be Extracted

Although calibrated RGB reconstruction cannot be obtained directly from event counts alone, controlled sequential RGB illumination enables useful **relative colour-response information** to be estimated.

These responses can be used to generate an ROI-level pseudo-colour representation of the event stream.

---

# Limitations

## Illumination Calibration

The red, green, and blue light sources may not produce identical optical power.

Differences in event responses may therefore be caused partly by differences in LED intensity rather than surface reflectance alone.

---

## Sensor Spectral Sensitivity

The monochrome event sensor may have different sensitivities at different wavelengths.

Quantitative colour reconstruction would require a calibrated model of the sensor's spectral response.

---

## Event Thresholds

Events are generated only when the change in logarithmic brightness exceeds the internal contrast threshold of the sensor.

Consequently, event density is not linearly proportional to absolute optical intensity.

---

## PWM Modulation

PWM-driven illumination introduces additional brightness transitions.

These transitions generate additional events and can make accurate temporal alignment more difficult.

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

# Future Work

## 1. Hardware Synchronisation

A direct synchronisation signal between the illumination controller and the event-camera recording system could provide precise illumination timestamps.

This would significantly reduce uncertainty during temporal alignment.

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

A calibrated spectral sensitivity model could then be included in the colour estimation process.

---

## 4. RGB Camera Ground Truth

A conventional RGB camera could be added to the experimental system to provide reference colour measurements.

The system could then compare:

```text
Event-derived colour estimate
            ↓
RGB camera ground truth
```

This would enable quantitative evaluation of reconstruction performance.

Possible metrics could include:

- RGB error,
- colour distance,
- reconstruction consistency,
- confusion between different surface colours.

---

## 5. Event + RGB Sensor Fusion

A longer-term extension is to combine the complementary properties of event cameras and conventional frame-based RGB cameras.

An event camera provides:

- high temporal resolution,
- low latency,
- reduced motion blur,
- high dynamic range.

An RGB camera provides:

- absolute intensity information,
- conventional colour information,
- dense spatial appearance.

Combining these two modalities could support more robust perception in challenging and highly dynamic environments.

---

## 6. Event + RGB + LiDAR Perception

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

## 7. Motion Compensation

Future work could investigate:

- event-based optical flow,
- feature tracking,
- motion estimation,
- object tracking,

to separate motion-generated events from illumination-generated events.

This would be necessary for extending the current controlled experiment toward real-world dynamic scenes.

---

# Repository Structure

The current and planned repository structure is:

```text
event-camera-colour-reconstruction/
│
├── README.md
├── .gitignore
│
├── requirements.txt
│
├── results/
│   │
│   ├── figures/
│   │   ├── experimental_setup.JPG
│   │   ├── flash_detection.png
│   │   ├── roi_response.png
│   │   └── rgb_response.png
│   │
│   └── demos/
│       └── pseudo_colour_result.gif
│
├── src/
│   ├── dat_decoder.py
│   ├── flash_detection.py
│   ├── temporal_alignment.py
│   ├── roi_analysis.py
│   └── colour_mapping.py
│
├── experiments/
│   └── experimental analysis and configuration
│
└── docs/
    └── technical_report.pdf
```

The source-code structure will continue to be organised as the experimental code is cleaned and documented for public release.

---

# Software

The project primarily uses:

- **Python**
- **NumPy**
- **OpenCV**
- **Matplotlib**

Additional dependencies will be listed in:

```text
requirements.txt
```

---

# Data Availability

The original event-camera recordings are not included in this public repository because of their large file size and data-sharing considerations.

Where appropriate, small example recordings may be added for demonstration or reproducibility.

No confidential, proprietary, or internally restricted research data are intended to be included in this repository.

---

# Technical Report

A detailed technical report describing the:

- experimental design,
- event-camera processing pipeline,
- illumination synchronisation strategy,
- ROI analysis,
- RGB response estimation,
- pseudo-colour reconstruction,
- experimental observations,
- limitations,
- future work,

will be provided under:

```text
docs/technical_report.pdf
```

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

The reported RGB response represents **relative ON-event behaviour under controlled sequential RGB illumination**.

The pseudo-colour output is intended for visualisation and should not be interpreted as calibrated surface RGB or conventional RGB video reconstruction.
