# Camera Calibration and Real-World Dimension Measurement

## Overview

This project demonstrates camera calibration using a checkerboard pattern and implements real-world dimension measurement of objects using perspective projection equations[web:13][web:15]. The system converts pixel coordinates to physical measurements using the pinhole camera model[web:13][web:18].

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Step 1: Prepare Calibration Pattern](#step-1-prepare-calibration-pattern)
- [Step 2: Capture Calibration Images](#step-2-capture-calibration-images)
- [Step 3: Camera Calibration](#step-3-camera-calibration)
- [Step 4: Save Calibration Parameters](#step-4-save-calibration-parameters)
- [Step 5: Real-World Dimension Measurement](#step-5-real-world-dimension-measurement)
- [Validation Experiment](#validation-experiment)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Features

- **Camera calibration** using checkerboard pattern[web:13][web:15]
- **Distortion correction** with lens distortion models[web:13][web:18]
- **Interactive ROI selection** for measurement regions[web:7]
- **Real-world dimension calculation** using pinhole camera model[web:18]
- **Validation experiments** with known objects[web:18]

---

## Prerequisites

### Hardware Requirements
- Camera (webcam, DSLR, or smartphone camera)
- Printed checkerboard pattern (8×6 or 9×6 recommended)[web:12][web:15]
- Flat surface for mounting the pattern[web:16]
- Object with known dimensions for validation[web:18]
- Measuring tape or ruler

### Software Requirements
