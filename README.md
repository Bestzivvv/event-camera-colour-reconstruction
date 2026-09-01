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

where:

x, y represent the pixel coordinates,
t represents the timestamp,
p represents the event polarity.

This sensing principle provides several advantages:

high temporal resolution,
low latency,
high dynamic range,
reduced motion blur,
sparse data representation.

However, a monochrome event camera does not directly provide RGB colour information.

This project explores whether relative colour information can be inferred by illuminating the scene sequentially with red, green, and blue light and analysing the corresponding event responses.

The overall processing pipeline is:

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

Research Questions

The project investigates the following questions:

How does a monochrome event camera respond to sequential red, green, and blue illumination?
Can event density measured under different illumination colours be used to estimate a relative RGB response?
How can the RGB illumination sequence be reliably synchronised with an asynchronous event stream?
How do ON and OFF events behave during controlled illumination changes?
What factors limit reliable colour reconstruction using event-camera measurements alone?
How could this approach be extended toward multimodal perception using event cameras, RGB cameras, and other sensing modalities?
