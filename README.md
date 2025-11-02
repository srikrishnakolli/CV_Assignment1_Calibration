# Camera Calibration and Real-World Dimension Measurement

## Project Overview

This project implements camera calibration using a checkerboard pattern from **calibdb.net** and develops a measurement system to calculate real-world dimensions of objects using perspective projection equations[web:13][web:15]. The system converts pixel coordinates to physical measurements through the pinhole camera model[web:13][web:26].

---


---

## Objectives

1. Calibrate a camera using checkerboard pattern from calibdb.net[web:13][web:15]
2. Extract camera intrinsic parameters (fx, fy, cx, cy) and distortion coefficients (k1, k2, k3, p1, p2)[web:13][web:26]
3. Implement pixel coordinate retrieval through mouse clicks[web:7]
4. Calculate real-world dimensions of objects using perspective projection[web:26]
5. Validate measurements with known object dimensions[web:18]

---

## Requirements

Python 3.7+
OpenCV 4.x

### Hardware
- Camera 
- Computer screen or printer for calibration pattern
- Objects with known dimensions for validation
- Measuring tape or ruler

### Software
