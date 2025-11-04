# Camera Calibration and Real-World Dimension Measurement

## Project Overview

This project implements camera calibration using a checkerboard pattern from **calibdb.net** and develops a measurement system to calculate real-world dimensions of objects using perspective projection equations. The system converts pixel coordinates to physical measurements through the pinhole camera model.

---



<img width="2716" height="1876" alt="image" src="https://github.com/user-attachments/assets/d6053b54-dfab-48e5-9cee-41d41613e128" />




---

## Objectives

1. Calibrate a camera using checkerboard pattern from calibdb.net
2. Extract camera intrinsic parameters (fx, fy, cx, cy) and distortion coefficients (k1, k2, k3, p1, p2)
3. Implement pixel coordinate retrieval through mouse clicks
4. Calculate real-world dimensions of objects using perspective projection
5. Validate measurements with known object dimensions

---

## Prerequisites
- Python 3.x
- OpenCV library
- Camera (webcam or external)
- Printer or computer screen for displaying calibration board
- Calibration board from calibdb.net



## Project Steps

### Step 1: Prepare Calibration Board
- Visit calibdb.net and download/print the calibration board
- Alternatively, display the board directly on a computer screen
- Ensure the board is flat and clearly visible

### Step 2: Capture Calibration Images
- Take at least 10 images of the calibration board using your camera
- Maintain the same camera orientation throughout
- Vary the distance and angles between the camera and board
- Ensure the entire board is visible in each image
- Use different perspectives for better calibration accuracy

### Step 3: Camera Calibration
- Run the provided Python code to calibrate your camera through the browser
- The calibration process will detect the board pattern in your images
- The algorithm will compute the camera's intrinsic parameters

### Step 4: Save Calibration Parameters
After calibration, note down the following values in a txt file:
- **Camera Matrix Parameters:**
  - fx (focal length x)
  - fy (focal length y)
  - cx (optical center x)
  - cy (optical center y)
- **Distortion Coefficients:**
  - k1, k2, k3 (radial distortion)
  - p1, p2 (tangential distortion)

### Step 5: Pixel Coordinate Retrieval
- Implement functionality to retrieve pixel coordinates of points clicked on an image
- Use a real-world object image for testing
- The code allows interactive point selection on the image

### Step 6: Real-World Dimension Measurement
**Task:** Write a script to find the real-world dimensions of an object using perspective projection equations.

**Implementation Steps:**
1. Capture an image of an object (e.g., ball, cube) from a known distance
2. Measure the actual distance between the camera and object accurately
3. Use the calibrated camera parameters (fx, fy, cx, cy, k1, k2, k3, p1, p2)
4. Apply perspective projection equations to calculate real-world dimensions
5. For a ball: calculate the diameter
6. For a cube: calculate the side length











# Results Images for the task


<table>
  <tr>
    <td> <img width="1037" height="751" alt="image" src="https://github.com/user-attachments/assets/96be2d3e-f3bd-4515-873a-28cdbbbddfa5" />
 </td>
    <td> <img width="875" height="532" alt="image" src="https://github.com/user-attachments/assets/a09cac9c-6c74-4816-931b-38395a4ab3a8" />
 </td>
  </tr>
</table>


