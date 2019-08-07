import numpy as np
import cv2
__author__ = 'David Arnaiz'

"""
Camera calibration.
This program is based on the tutorial from:
"https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html"

For the calibration process a chessboard image with a grid bigger than 6 by 7 squares is required. You can use the "Chessboard.png" image in this folder. For the calibration process first the camera is going to check for frames with a valid pattern and once is found the corners are stored in a list. Then the program will stop searching for the pattern for 5 seconds so it can be repositioned correctly, while its waiting the frame with the corners will be shown to check if the corners are correctly positioned and that the lines between two corners match the chessboard grid. When 15 frames are correctly analysed the camera will stop and the parameters will be calculated. To take the images its better to keep the chessboard image still and move the camera or keep the camera still. It also requires a white background for the pattern to have a robust calibration. Once the model is calculated it will be stored in the file 'camera_model.mdl' for future reference as the camera model only has to be obtained once. ms = mean squared
"""


def printmodel(cam_matrix, rms_error, distortion):  # Prints the model in a .mdl file for future use
    """ The model is composed by the camera matrix (3x3 matrix), the error MS value (single value) and the distortion     vector (1x5 vector). they will be stored in a file in the following way:
        - The camera matrix: a string header "camera matrix" followed by the values of each row in order.
        - The error MS value: a string header "Error MS", then the value.
        - The distortion vector: a string header "distortion vector" followed by the values in order.
    """
    file2 = open('camera_model.mdl', 'w')  # open/create file to write
    # Write the camera matrix
    file2.write("camera matrix" + '\n')
    for g in cam_matrix:
        for gg in g:
            file2.write(str(gg) + '\n')
    # Write the MS error
    file2.write("MS error" + '\n')
    file2.write(str(ms_error) + '\n')
    # Write the distortion matrix
    file2.write("distortion" + '\n')
    for g in distortion:
        for gg in g:
            file2.write(str(gg) + '\n')
    file2.close()
    print("The camera model was successfully saved")


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) in order of coordinates
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# If left like this the parameters will be found in unit of the size of the blocks conforming the chessboard, in order
# to have the parameters in mm it has to be multiplied by the size of the blocks. In this case the block are 26x26 mm
objp *= 2.6  # Multiply each coordinate by the size of the blocks to get the distance in cm (2.6 cm)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# Image acquisition
"""
In this process the camera will start recording looking for the pattern. Once the pattern is found it will show the image until any key is pressed, if the pressed key is 'c' it will cancel the image, if the key is 'q' at any time the program will end, if any other key is pressed then it will save the captured image (frame_n.png), the image with the corners painted (corners_n.png) for future reference and store the corners in the matrix objpoints (real 3d points) and imgpoints (image points). It will repeat this process until 15 valid images are found. In order to get a good result the pattern has to be in different positions respect to the camera and while finding the pattern the image has to be still respect to the camera to get a good image, also check that the corners found are correct and the pattern is well detected.
"""
n = 0  # number of images
cap = cv2.VideoCapture(0)  # Select the camera
while n < 15:
    ret, frame = cap.read()  # capture frame ret --> boolean to check if the frame is correct
    if ret:  # check if the frame is correctly read
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gframe, (7, 6), None)
        if ret:  # if the pattern is found
            # Refine the corners
            corners2 = cv2.cornerSubPix(gframe, corners, (11, 11), (-1, -1), criteria)
            # Draw the corners
            frame = cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)
            # Show image
            cv2.imshow('frame', frame)
            print('Pattern found',  str(n+1), 'of 15')
            print("Press 'c' to cancel, 'q' to exit or any key to continue")
            k = cv2.waitKey(0)
            if k == ord('c'):
                print('canceled')
            elif k == ord('q'):
                break
            else:
                n += 1
                objpoints.append(objp)
                imgpoints.append(corners2)
                cv2.imwrite('frame_'+str(n)+'.png', gframe)
                cv2.imwrite('corners_'+str(n)+'.png', frame)
                cv2.waitKey(500)
            # Capture some frames to avoid getting the buffered images
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
            ret, frame = cap.read()
        else:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    else:
        print('Error couldn\'t read frame')
        break
cv2.destroyAllWindows()
cap.release()
if n < 15:
    print('Not enough images ', str(n), 'of 15')
    exit()


# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gframe.shape[::-1], None, None)
"""
ret: is a binary variable indicating if the camera was successfully calibrated

mtx: is the camera matrix based on the focal lengths and the optical centre.

dist: is the distortion matrix that can be used to correct image distortion.

rvecs: is a matrix of vectors indicating the rotation values of the 15 images.

tvects: is a matrix of vector indicating the translation values of the 15 images.
"""
if ret:
    """
    Calculate de projection error. To estimate de error in the found parameters. First the object points(3D estimation points) are transformed into image points using the found parameters. Then the error is the norm (módulo del vector) between the real image points found by the cv2.findChessBoardCorners() function and the projected ones using the found parameters, so the error is estimated for all 42 points of each image. the error is expressed as maximum error, MS error and mean error.
    """
    error = []  # Store the error values for every picture
    for i in range(len(objpoints)):
        projtpts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)  # project the 3D estimated points
        error.append(cv2.norm(imgpoints[i], projtpts, cv2.NORM_L2))  # Calculate the norm --> error
        # cv2.NORM_L2 calculates the norm as euclidean (square root of the sum of the square of the numbers)
        # distance not manhattan distance

    # Print the results
    # Calibration parameters
    print('the camera was successfully calibrated¡¡')
    print('........................................')
    print('The camera matrix is:')
    print(mtx)
    print('The camera distortion is:')
    print(dist)
    # Error parameters
    print("the MS error is:")
    MS = 0
    for i in error:
        RMS += i*i
    RMS /= len(error)
    print(MS)
    print("The mean error in mm is:")
    mean_error = sum(error)/len(error)
    print(mean_error)

    # Save camera model to a file
    # First confirm if the camera model wants to be saved
    while True:
        print("Do you want to save the camera model?")
        print("insert 'y' or 'yes' to save it, 'n' or 'not' to cancel")
        try:
            inp = str(input('>')).lower()
        except ValueError:
            print("Error, invalid input. Try again.      E1")
            continue
        if inp == 'y' or inp == 'yes':  # The model wants to be saved
            try:  # check if the file exists
                file = open('camera_model.mdl', 'r')
                file.close()
                while True:  # The file exists confirm if it wants to be overwritten
                    try:
                        print("There is already a camera model. Do you want to overwrite it?")
                        print("Insert 'y' or 'yes' to overwrite, 'n' or 'no' to cancel")
                        inp = str(input('>')).lower()
                    except ValueError:
                        print("Error, invalid input. Try again.     E2")
                        continue
                    if inp == 'y' or inp == 'yes':  # The file wants to be overwritten
                        printmodel(mtx, MS, dist)
                        break
                    elif inp == 'n' or inp == 'no':
                        print("The model wont be saved")
                        break
                    else:
                        print("Unknown command please try again.    E3")
                        continue
            except IOError:  # The file doesn't exist create new
                printmodel(mtx, RMS, dist)
            break
        elif inp == 'n' or inp == 'no':  # The camera model doesn't want to be saved
            while True:
                try:
                    print("Cancel saving the camera model?")
                    print("Insert 'y' or 'yes' to overwrite, 'n' or 'no' to cancel")
                    inp = str(input('>')).lower()
                    if inp == 'y' or inp == 'yes':  # Don't save the camera model
                        inp = 'false'
                        break
                    elif inp == 'n' or inp == 'no':  # Not sure
                        inp = 'true'
                        break
                    else:
                        print("Unknown command please try again.    E4")
                        continue
                except ValueError:
                    print("Error, invalid input. Try again.     E5")
                    continue
            if inp == 'false':
                break
            else:
                continue
        else:
            print("Unknown command please try again.    E6")
            continue
else:
    print("The camera couldn't be calibrated")
