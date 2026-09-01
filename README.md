# Event-Based Colour Response Analysis and Pseudo-Colour Reconstruction

This repository contains the code, experimental workflow, and results from my 2026 summer research project at the University of Manchester.

The project investigates whether relative colour information can be extracted from a monochrome event camera by combining asynchronous event measurements with sequential red, green, and blue illumination.

The main focus is on event-stream processing, temporal synchronisation, ROI-based event-response analysis, and pseudo-colour reconstruction.

---

## Overview

Event cameras record changes in logarithmic brightness asynchronously rather than capturing conventional image frames at fixed intervals. This provides several advantages, including high temporal resolution, low latency, high dynamic range, and reduced motion blur.

However, a monochrome event camera does not directly provide RGB colour information.

This project explores a possible way of estimating relative colour responses by illuminating a scene sequentially with red, green, and blue light and analysing the event-camera response produced during each illumination phase.

The overall pipeline is:

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

---

# Event-Based Colour Response Analysis and Pseudo-Colour Reconstruction

This project investigates several questions:

How does a monochrome event camera respond to sequential RGB illumination?
Can the event density generated under different illumination colours be used to estimate a relative RGB response?
How can illumination phases be aligned reliably with asynchronous event streams?
How do ON and OFF events behave under controlled flashing illumination?
What are the main limitations of recovering colour information using an event-only sensing pipeline?
How could the method be extended toward multimodal event-camera and frame-camera perception?
