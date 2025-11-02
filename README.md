# CSc 8830: Computer Vision : Assignment 1 Solutions

## Question 1

To calibrate the camera, I have used a 8x6 checkerboard pattern.

The code to take 10 images of the checker board using both the monochrome (left and right) and color cameras can be found in the file ques1.ipynb The images captured can found in the folder 'images'.

The images are then used for calculating the camera matrix, translation and rotational vectors.

The corrected images are found in the 'images' folder and the corresponding camera values are stored in their respective folders (left, right and color).

<table>
  <tr>
    <td><img src="images/left/17109820135253.png" alt="Image 1"></td>
    <td><img src="images/left/17109820135253_corners.png" alt="Image 2"></td>
    <td><img src="images/left/17109820135253_result.png" alt="Image 3"></td>
  </tr>
</table>

___

## Question 2

Used the camera matrix, rotational matrix, and translation matrix (previously derived in Question 1) to convert image coordinates to world coordinates.

Distance Measurement in 3D Space: Selected two pixel coordinates from an image, converted them into world coordinates using the transformation matrices, and calculated the distance between these 3D points, determining the real-world distance.

Successfully computed the rotation matrix from the extrinsic parameters and subsequently calculated the rotation angles along each axis, choosing the order of rotation based on the experimentation setup and requirements.

___

## Question 3

I designed and implemented a script utilizing perspective projection equations to determine the real-world dimensions of objects.

The implementation includes:

- **ROI Selection**: Resizable window with interactive ROI selection on the display image
- **Coordinate Mapping**: Maps ROI from display coordinates back to original image coordinates with proper scaling
- **Camera Intrinsics**: Uses calibrated camera parameters (FX, FY, cx, cy) for accurate measurements
- **Distance Calculation**: Computes real-world width and height using pinhole camera model with measured distance Z

### Camera Parameters Used:
