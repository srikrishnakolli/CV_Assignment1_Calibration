# Camera Calibration ➜ Real-World Dimension Estimation

This module has two parts:

Camera Calibration – print a calibration board, capture images, and compute your camera intrinsics and distortion.

Real-World Measurement – click/drag an ROI on a scene image and compute the object’s true width/height using perspective projection.

Everything runs locally on Windows/macOS/Linux with Python.



Requirements

Python 3.9+

Install once:

pip install -U opencv-python numpy matplotlib

pip install pillow


<img width="2716" height="1876" alt="image" src="https://github.com/user-attachments/assets/dfdc1040-07da-4eb0-a00d-b64175f1d985" />





Part 1 — Camera Calibration
Step 1 — Get a calibration board

Download/print a checkerboard from calibdb.net (or any standard source).

Recommended inner-corner grid (cols × rows): 9×6 (or 7×6).

Mount to a flat, rigid surface (foam board/glass).

Measure square size precisely (e.g., 24.0 mm).

Image placeholder — replace later:


Step 2 — Capture images

Use the same camera you’ll measure with later.

Take ≥ 10 photos from the same camera orientation but vary distance and angles:

Change tilt/rotation/position in the frame; include corners & edges.

Avoid motion blur; use even lighting and sharp focus.

Image placeholder — replace later:


Step 3 — Calibrate with Python

Use your course calibration code (or any OpenCV chessboard calibration) to compute K and distortion.

Windows (PowerShell)

python .\calibrate.py --images ".\Calibration\*.JPG" --grid 9x6 --square 24.0 --out calibration.npz --save-report


macOS / Linux

python ./calibrate.py --images "./Calibration/*.JPG" --grid 9x6 --square 24.0 --out calibration.npz --save-report


Should produce:

Camera matrix K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]

Distortion coefficients k1, k2, p1, p2, k3









Annotatted Image of the Object:

<img width="1523" height="850" alt="image" src="https://github.com/user-attachments/assets/b56afd00-b6c6-457d-9a81-759eabd609fa" />
