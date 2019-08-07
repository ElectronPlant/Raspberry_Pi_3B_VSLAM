import cv2
import numpy as np

# Markers
"""
marker1 = np.array([[0, 0, 255], [255, 0, 255], [0, 255, 0]])
marker2 = np.array([[255, 0, 255], [0, 255, 0], [255, 0, 0]])
marker3 = np.array([[0, 255, 0], [0, 0, 0], [255, 255, 255]])
marker4 = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]])
marker5 = np.array([[255, 0, 255], [0, 255, 0], [255, 255, 255]])
marker_position = np.array([[[57.5, -10.6, 32.4, 1], [57.5, -17.8, 32.4, 1],
                              [57.5, -10.6, 26.1, 1], [57.5, -10.6, 26.1, 1]],
                             [[57.5, -10.6, 17.3, 1], [57.5, -16.8, 17.2, 1],
                              [57.5, -10.6, 11, 1], [57.5, -16.8, 11, 1]],
                             [[57.5, -10.6, 32.4, 1], [57.5, -17.8, 32.4, 1],
                              [57.5, -10.6, 26.1, 1], [57.5, -10.6, 26.1, 1]],
                             [[57.5, -10.6, 32.4, 1], [57.5, -17.8, 32.4, 1],
                              [57.5, -10.6, 26.1, 1], [57.5, -10.6, 26.1, 1]],
                             [[57.5, -10.6, 32.4, 1], [57.5, -17.8, 32.4, 1],
                              [57.5, -10.6, 26.1, 1], [57.5, -10.6, 26.1, 1]]])
"""
marker1 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
marker2 = np.array([[0, 255, 0], [255, 0, 0], [0, 255, 255]])
marker3 = np.array([[255, 255, 255], [0, 255, 0], [255, 0, 255]])
marker4 = np.array([[255, 0, 255], [0, 255, 0], [255, 0, 0]])
marker5 = np.array([[0, 255, 255], [255, 0, 255], [255, 255, 0]])
marker6 = np.array([[255, 0, 255], [0, 255, 0], [255, 0, 255]])
marker7 = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]])
marker8 = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 255]])

"""
marker_position = np.array([[[95.5, -90.8, 22.1], [95.5, -90.8, 12.1],
                             [85.5, -90.8, 12.1], [85.5, -90.8, 22.1]],
                            [[114.5, -90.8, 17.5], [114.5, -90.8, 7.5],
                             [104.5, -90.8, 7.5], [104.5, -90.8, 17.5]],
                            [[148.3, 5, 13.2], [148.3, 5, 3.2],
                             [148.3, -5, 3.2], [148.3, -5, 13.2]],
                            [[-32.4, -55, 15.8], [-32.4, -55, 5.8],
                             [-32.4, -45, 5.8], [-32.4, -45, 15.8]]])

known_landmarks = [1, 2, 3, 4]
landmark_covariance = np.array(np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                               np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                               np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                               np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
"""
marker_position = np.array([[[95.5, -88.8, 22.1], [95.5, -88.8, 12.1],
                             [85.5, -88.8, 12.1], [85.5, -88.8, 22.1]]])
known_landmarks = [1]
landmark_covariance = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
lc_deff = True
landmark_robot_covariance = np.matrix([[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]])
lrc_deff = True
robot_covariance = np.matrix([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
p = robot_covariance

# Inverse measure model noise
u = np.identity(3)*417.105724158432*100
unknown_landmarks = []
unknown_landmarks_data = []

# camera measurements
xm = 6
ym = -2
zm = 13


def identify_marker(marker_matrix):  # Identifies the marker given the marker matrix
    if np.array_equal(marker_matrix, marker1):
        marker = 1
    elif np.array_equal(marker_matrix, marker2):
        marker = 2
    elif np.array_equal(marker_matrix, marker3):
        marker = 3
    elif np.array_equal(marker_matrix, marker4):
        marker = 4
    elif np.array_equal(marker_matrix, marker5):
        marker = 5
    elif np.array_equal(marker_matrix, marker6):
        marker = 6
    elif np.array_equal(marker_matrix, marker7):
        marker = 7
    elif np.array_equal(marker_matrix, marker8):
        marker = 8
    else:
        marker = 0
    return marker


def read_camera_model():  # This function will read the camera model named "cameramodel.mdl"
    """
    :Input --> nothing.
    :Output --> camera model matrix.
    :Output --> calibration MS error (supposed the variation of the gaussian error distribution).
    :Output --> distortion values (used to correct the images that are been processed)
    :Output --> a binary value set to True if the camera model file was read successfully.
    This function reads the camera_model.mdl in the same folder as this file. The camera model file should be generated
    by the program 'camera_calibration.py', or have a compatible structure.
    """

    # Read the file
    # First check if the file exists
    try:
        # the file exists
        file = open('camera_model.mdl', 'r')  # Read the file
        # Read the camera matrix
        inp = file.readline()
        inp2 = ''
        for i in inp:
            if i != '\n':
                inp2 = str(inp2 + i)
            else:
                break
        if inp2 == 'camera matrix':
            mtx = np.zeros((3, 3), np.float64)  # Predefine matrix
            for i in range(0, 9):
                inp = file.readline()
                inp2 = ''
                for j in inp:
                    if j != '\n':
                        inp2 = str(inp2 + j)
                    else:
                        break
                if i < 3:
                    mtx[0, i] = float(inp2)
                elif i < 6:
                    mtx[1, i-3] = float(inp2)
                else:
                    mtx[2, i-6] = float(inp2)
        else:
            print('Error reading the camera model file.    1')
            return '', '', '', False
        # Read the MS error
        inp = file.readline()
        inp2 = ''
        for i in inp:
            if i != '\n':
                inp2 = str(inp2 + i)
            else:
                break
        if inp2 == 'MS error':
            inp = file.readline()
            inp2 = ''
            for i in inp:
                if i != '\n':
                    inp2 = str(inp2 + i)
                else:
                    break
            rms_error = float(inp2)
            # MS error value successfully read
        else:
            print("Error reading the camera model file.    2")
            return '', '', '', False
        # Reading the distortion values
        inp = file.readline()
        inp2 = ''
        for i in inp:
            if i != '\n':
                inp2 = str(inp2 + i)
            else:
                break
        if inp2 == 'distortion':
            distortionv = np.zeros((1, 5), np.float64)  # Predefine matrix
            for i in range(0, 5):
                inp = file.readline()
                inp2 = ''
                for j in inp:
                    if j != '\n':
                        inp2 = str(inp2 + j)
                    else:
                        break
                distortionv[0, i] = float(inp2)
            return mtx, rms_error, distortionv, True
        else:
                print('Error reading the camera model file.    3')
                return '', '', '', False
    except IOError:
        print('Error, no camera model file found, please use camera_calibration.py to generate one')
        return '', '', '', False


def obtain_landmark_pose(pos_im1, pos_im2, T1, T2, mtx):  # This function obtains the global landmarks pose
    """
    :Input --> pos_im1: Pixel coordinates of the landmarks from the first image (numpy.ndarray).
    :Input --> pos_im2: Pixel coordinates of the corresponding landmarks from the second image (same length and type as im1_pos)
    :Input --> T1: Rotation and translation matrix of the camera in the first position.
    :Input --> T2: Rotation and translation matrix of the camera in the second position.
    :Input --> mtx: Camera matrix.
    :Output --> pos: Landmarks pose as global coordinates (array).
    From the image only the pixel coordinates can be obtained (u,v) to define the global landmark position we need three
    coordinates (x,y,z), so we can't obtain the landmark position just from one image. The global coordinates of the
    landmark can be obtained from two images if the corresponding pixel coordinates can be found. This function obtains
    the global coordinates of the landmarks given the camera position when the two images were taken.
    """
    # A*global_pose=B --> global_pose = inv(At*A)*At*B
    pos = np.zeros((len(pos_im1[0]), 3), np.float64)
    print('u1: ' + str(pos_im1))
    print('v1: ' + str(pos_im2))
    for i in range(0, len(pos_im1[0])):
        u1 = pos_im1[0][i].item(0)
        v1 = pos_im1[0][i].item(1)
        u2 = pos_im2[0][i].item(0)
        v2 = pos_im2[0][i].item(1)
        a = np.matrix([[(T1[2,0]*((u1-mtx[0,2])/mtx[0,0]))-T1[0,0], (T1[2,1]*((u1-mtx[0,2])/mtx[0,0]))-T1[0,1], (T1[2,2]*((u1-mtx[0,2])/mtx[0,0]))-T1[0,2]],
                      [(T1[2,0]*((v1-mtx[1,2])/mtx[1,1]))-T1[1,0], (T1[2,1]*((v1-mtx[1,2])/mtx[1,1]))-T1[1,1], (T1[2,2]*((v1-mtx[1,2])/mtx[1,1]))-T1[1,2]],
                      [(T2[2,0]*((u2-mtx[0,2])/mtx[0,0]))-T2[0,0], (T2[2,1]*((u2-mtx[0,2])/mtx[0,0]))-T2[0,1], (T2[2,2]*((u2-mtx[0,2])/mtx[0,0]))-T2[0,2]],
                      [(T2[2,0]*((v2-mtx[1,2])/mtx[1,1]))-T2[1,0], (T2[2,1]*((v2-mtx[1,2])/mtx[1,1]))-T2[1,1], (T2[2,2]*((v2-mtx[1,2])/mtx[1,1]))-T2[1,2]]])
        b = np.matrix([[T1[0,3]-(T1[2,3]*((u1-mtx[0,2])/mtx[0,0]))],
                      [T1[1,3]-(T1[2,3]*((v1-mtx[1,2])/mtx[1,1]))],
                      [T2[0,3]-(T2[2,3]*((u2-mtx[0,2])/mtx[0,0]))],
                      [T2[1,3]-(T2[2,3]*((v2-mtx[1,2])/mtx[1,1]))]])
        # Solve the equation if not consider to be in origin
        try:
            coordinates = np.dot(np.linalg.inv(np.dot(a.transpose(), a)), np.dot(a.transpose(), b))
            pos[i] = coordinates.transpose()
        except (RuntimeError, TypeError, NameError, np.linalg.linalg.LinAlgError):
            pos[i] = np.matrix([0, 0, -1])
            print('pos' + str(pos))
    return pos


def landmark_pose(pos1, im1, pos2, im2, pos3, im3):
    d_pose = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    n = [0, 0, 0, 0]
    t1 = np.matrix([[np.sin(pos1.item(2)), -np.cos(pos1.item(2)), 0,
                     ym + pos1.item(1)*np.cos(pos1.item(2)) -
                     pos1.item(0)*np.sin(pos1.item(2))],
                    [0, 0, -1, zm],
                    [np.cos(pos1.item(2)), np.sin(pos1.item(2)), 0,
                     -(pos1.item(0)*np.cos(pos1.item(2)) +
                       pos1.item(1)*np.sin(pos1.item(2))) - xm],
                    [0, 0, 0, 1]])
    t2 = np.matrix([[np.sin(pos2.item(2)), -np.cos(pos2.item(2)), 0,
                     ym + pos2.item(1)*np.cos(pos2.item(2)) -
                     pos2.item(0)*np.sin(pos2.item(2))],
                    [0, 0, -1, zm],
                    [np.cos(pos2.item(2)), np.sin(pos2.item(2)), 0,
                     -(pos2.item(0)*np.cos(pos2.item(2)) +
                       pos2.item(1)*np.sin(pos2.item(2))) - xm],
                    [0, 0, 0, 1]])
    t3 = np.matrix([[np.sin(pos3.item(2)), -np.cos(pos3.item(2)), 0,
                     ym + pos3.item(1)*np.cos(pos3.item(2)) -
                     pos3.item(0)*np.sin(pos3.item(2))],
                    [0, 0, -1, zm],
                    [np.cos(pos3.item(2)), np.sin(pos3.item(2)), 0,
                     -(pos3.item(0)*np.cos(pos3.item(2)) +
                       pos3.item(1)*np.sin(pos3.item(2))) - xm],
                    [0, 0, 0, 1]])
    d_pose1 = obtain_landmark_pose(im1, im2, t1, t2, matrix)
    for j in range(0, len(d_pose1)):
        if d_pose1[j][2] >= 0:
            d_pose[j] = [int(d_pose[j][0]) + int(d_pose1[j][0]),
                         int(d_pose[j][1]) + int(d_pose1[j][1]),
                         int(d_pose[j][2]) + int(d_pose1[j][2])]
            n[j] += 1
    d_pose2 = obtain_landmark_pose(im1, im3, t1, t3, matrix)
    for j in range(0, len(d_pose2)):
        if d_pose2[j][2] >= 0:
            d_pose[j] = [int(d_pose[j][0]) + int(d_pose2[j][0]),
                         int(d_pose[j][1]) + int(d_pose2[j][1]),
                         int(d_pose[j][2]) + int(d_pose2[j][2])]
            n[j] += 1
    d_pose3 = obtain_landmark_pose(im2, im3, t2, t3, matrix)
    for j in range(0, len(d_pose3)):
        if d_pose3[j][2] >= 0:
            d_pose[j] = [int(d_pose[j][0]) + int(d_pose3[j][0]),
                         int(d_pose[j][1]) + int(d_pose3[j][1]),
                         int(d_pose[j][2]) + int(d_pose3[j][2])]
            n[j] += 1
    if 0 in n:
        return False, [0, 0, 0]
    else:
        for j in range(0, len(d_pose)):
            for jj in range(0,3):
                d_pose[j][jj] = int(d_pose[j][jj])/n[j]
    return True, d_pose


def pattern_filter(image):  # detects the patterns if any in the image.
    """
    ------- Consider just considering the pixels connected to the main floor-----
    :Input  --> image: cv2 RGB image (480 x 640)
    :Input  --> mask: Binary image from the previous colorFilter() step
    :Output --> mask: Binary image with white filter were the floor has been detected and black in the rest
    This function completes the previous step on filtering the floor. In this case the contour of the obstacles is
    detected and all the pixels on top aren't considered as part of the floor. Then the mask is altered to make dose
    pixels black (not considered floor).
    """
    correct = False
    cc = 0
    c_points = 0
    n_marker = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change the image to gray scale
    # Noise filtering
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur to take the noise out
    # Adaptative threshold
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    # Find contours
    edges = cv2.Canny(th2, 200, 20, 3)  # Canny edge detection algorithm to find the edges
    im1, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Erase non marks

    kk = 0
    contours3 = contours
    for i in range(len(contours)):
        if len(contours[i]) >= 4:  # Not consider shapes with less than 4 corners
            perimeter = cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], 0.03*perimeter, True)
            if len(approx) == 4:  # the approximation gives a four point shape (rectangle)
                area = cv2.contourArea(approx)
                pos, rect_area, angle = cv2.minAreaRect(approx)
                # x,y,w,h = cv2.boundingRect(contours[i])
                rect_area = rect_area[1]*rect_area[0]
                if not rect_area == 0:
                    extent = float(area)/rect_area
                else:
                    extent = 0
                if 0.8 < extent < 1.2 and 350 < rect_area:  # Area close to the rectangle one and big enough
                    contours3[kk] = approx
                    kk += 1
                else:
                    contours3 = np.delete(contours3, kk, 0)
            else:
                contours3 = np.delete(contours3, kk, 0)
        else:
            contours3 = np.delete(contours3, kk, 0)

    # Erase inner contours
    kk = 0
    contours2 = contours3
    for i in range(len(contours3)-1):
        (x, y), radius = cv2.minEnclosingCircle(contours3[i])
        (x2, y2), radius = cv2.minEnclosingCircle(contours3[i+1])
        if -10 < x-x2 < 10 and -10 < y-y2 < 10:
            contours2 = np.delete(contours2, kk + 1, 0)
        else:
            kk += 1

    # detect marker

    pts1 = np.float32([[0, 0], [0, 100], [100, 100]])  # target points for the affine transform
    # ppts1 = np.float32([[0, 0], [0, 100], [100, 100], [100, 0]])
    for cont in contours2:
        c = (cont[0, 0, :] + cont[1, 0, :] + cont[2, 0, :] + cont[3, 0, :])/4
        if cont[0, 0, 0] <= c[0]:
            pts2 = np.float32([cont[0, 0, :], cont[1, 0, :], cont[2, 0, :]])  # marker points in the image
            # ppts2 = np.float32([cont[0, 0, :], cont[1, 0, :], cont[2, 0, :], cont[3, 0, :]])
            points = np.array([[np.transpose(cont[0, 0, :])], [np.transpose(cont[1, 0, :])],
                               [np.transpose(cont[2, 0, :])], [np.transpose(cont[3, 0, :])]])
        else:
            pts2 = np.float32([cont[1, 0, :], cont[2, 0, :], cont[3, 0, :]])  # marker points in the image
            # ppts2 = np.float32([cont[1, 0, :], cont[2, 0, :], cont[3, 0, :], cont[0, 0, :]])
            points = np.array([[np.transpose(cont[1, 0, :])], [np.transpose(cont[2, 0, :])],
                               [np.transpose(cont[3, 0, :])], [np.transpose(cont[0, 0, :])]])

        # Affine transform
        m = cv2.getAffineTransform(pts2, pts1)
        # mm = cv2.getPerspectiveTransform(ppts2, ppts1)
        marker = cv2.warpAffine(gray, m, (100, 100))
        # marker = cv2.warpPerspective(gray, mm, (100, 100))
        # threshold image
        ret3, marker = cv2.threshold(marker, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # check marker
        if marker[10, 10] == 0 and marker[90, 10] == 0 and marker[90, 90] == 0 and marker[10, 90] == 0:
            mmark = np.matrix([[marker[30, 30], marker[30, 50], marker[30, 70]],
                              [marker[50, 30], marker[50, 50], marker[50, 70]],
                              [marker[70, 30], marker[70, 50], marker[70, 70]]])
            """
            # Try to calculate angle also-----------------------------------------------------------------------------------
            # angle calculation assuming that the marker is perfectly perpendicular to the floor
            x,y,w,h = cv2.boundingRect(cont)
            if w < h:
                theta = np.arccos(w/h)
            else:
                theta = 0
            print(theta*180/np.pi)
            """
            """
            # For printing only
            if np.array_equal(mmark, marker1):
                image = cv2.polylines(image, [cont], True, (0, 255, 0), 3)
                image = cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0), -1)
                cv2.putText(image, str(c), (int(round(c[0] + 10)), int(round(c[1]))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 255, 255), 2, cv2.LINE_AA)
            elif np.array_equal(mmark, marker2):
                image = cv2.polylines(image, [cont], True, (255, 0, 255), 3)
                image = cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (255, 0, 255), -1)
                cv2.putText(image, str(c), (int(round(c[0] + 10)), int(round(c[1]))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 0), 2, cv2.LINE_AA)
            """
            num = identify_marker(mmark)
            if num > 0:
                if correct:
                    cc = np.append(cc, np.matrix([c]), axis=0)
                    c_points = np.append(c_points, points, axis=0)
                    n_marker = np.append(n_marker, np.matrix(num), axis=0)
                else:
                    correct = True
                    cc = np.matrix([c])
                    c_points = points
                    n_marker = np.matrix(num)
    return correct, n_marker, cc, c_points


def pattern(fframe, covariance):  # Deals with the pattern detection
    ret, marker, center, points = pattern_filter(fframe)
    ret2 = False
    new = []
    newp = []
    mmarker = []
    if ret:  # landmark detected
        # check if the landmarks are known
        k = 0
        kk = True
        for i in range(0, len(marker)):
            # known landmark
            if marker.item(i) in known_landmarks:
                mmarker.append(marker.item(i))
                index = known_landmarks.index(marker.item(i))
                # known landmark
                r_center = marker_position[index]
                r_center = np.reshape(r_center, (4, 3))
                if k == 0:
                    rcenter = np.matrix(r_center)
                    ret2 = True
                else:
                    rcenter = np.vstack([rcenter, r_center])
                for j in range(0, 4):
                    if kk:
                        p_points = np.transpose(points[i*4+j])
                        kk = False
                    else:
                        p_points = np.vstack([p_points, np.transpose(points[i*4+j])])
                k += 1
            # Unknown landmark
            else:
                # Unknown landmarks
                new.append(marker.item(i))
                newp.append(points)
        lmarker = k
        # Just representation
        for i in points:
            fframe = cv2.circle(fframe, (int(round(i.item(0))), int(i.item(1))), 3, (255, 100, 100), -1)
            cv2.putText(fframe, '(' + str(float(i.item(0))) + ',' +
                        str(float(i.item(1))) + ')', (int(round(i.item(0)-10)), int(round(i.item(1) + 20))),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (20, 100, 20), 1, cv2.LINE_AA)
        if not ret2:
            lmarker = 0
            rcenter = 0
            p_points = 0
    else:
        lmarker = 0
        rcenter = 0
        p_points = 0
    cv2.imshow('frame', fframe)
    cv2.imwrite('markksss.png', fframe)

    return ret2, lmarker, p_points, rcenter, fframe, new, newp, mmarker


def add_landmarks(pos, covariance, new, newp, old):
    global marker_position, landmark_robot_covariance, landmark_covariance, lrc_deff, lc_deff
    # Check the number of times the landmark has been seen
    for i in new:
        if i in unknown_landmarks:
            # second or third time hte landmark has been seen
            index = unknown_landmarks.index(i)
            print('ppp' + str(unknown_landmarks_data[index][0]))
            if unknown_landmarks_data[index][0]:
                # Third time the landmark has been seen
                pos1 = unknown_landmarks_data[index][4][1]
                dist = np.sqrt((pos.item(0) - pos1.item(0))**2 +
                               (pos.item(1) - pos1.item(1))**2)
                print(dist)
                if dist >= 20:
                    pos1 = unknown_landmarks_data[index][2]
                    pos2 = unknown_landmarks_data[index][4][1]
                    im1 = unknown_landmarks_data[index][1]
                    im2 = unknown_landmarks_data[index][4][0]
                    if len(new)+ len(old) == 1:
                        im3 = newp
                    else:
                        inndex = new.index(i)*4
                        im3 = np.array([newp[0][inndex:inndex+4]])
                    print(1)
                    calc, d_pose = landmark_pose(pos1, im1, pos2, im2, pos, im3)
                    if calc:

                        # Add landmark
                        # Add to known landmark
                        known_landmarks.append(i)
                        inndex = new.index(i)*4
                        if lc_deff:
                            marker_position = np.vstack([marker_position, np.array([d_pose])])
                        else:
                            lc_deff = True
                            marker_position = np.array([d_pose])
                        """
                        # Add covariance
                        # X
                        covariance1 = np.matrix(unknown_landmarks_data[index][3])
                        covariance2 = np.matrix(unknown_landmarks_data[index][4][2])
                        pos1 = unknown_landmarks_data[index][2] +\
                               np.matrix([[1],
                                          [0],
                                          [0]]) * covariance1.item(0)
                        pos2 = unknown_landmarks_data[index][4][1] + \
                               np.matrix([[1],
                                          [0],
                                          [0]]) * covariance2.item(0)
                        pos3 = pos + np.matrix([[1],
                                                [0],
                                                [0]]) * covariance.item(0)
                        print(2)
                        calc, dd_pose = landmark_pose(pos1, im1, pos2, im2, pos, im3)
                        if calc:
                            dx_x = 0
                            dx_y = 0
                            dx_z = 0
                            for l in range(0, len(dd_pose)):
                                dx_x += abs(int(dd_pose[l][0]) - int(int(d_pose[l][0])))*0
                                dx_y += abs(int(dd_pose[l][1]) - int(int(d_pose[l][1])))*0
                                dx_z += abs(int(dd_pose[l][2]) - int(int(d_pose[l][2])))*0
                            ax = np.mean([covariance1.item(0) + covariance2.item(0) + covariance.item(0)])
                            dx_x /= (4*ax)
                            dx_y /= (4*ax)
                            dx_z /= (4*ax)
                            print('resx: ' + str([dx_x, dx_y, dx_z]))
                        else:
                            # one of the points can't estimate it
                            dx_x = 20000
                            dx_y = 20000
                            dx_z = 20000
                            print(':(1')
                        # Y
                        pos1 = unknown_landmarks_data[index][2] +\
                               np.matrix([[0],
                                          [1],
                                          [0]]) * covariance1.item(1)
                        pos2 = unknown_landmarks_data[index][4][1] + \
                               np.matrix([[0],
                                          [1],
                                          [0]]) * covariance2.item(1)
                        pos3 = pos + np.matrix([[0],
                                                [1],
                                                [0]]) * covariance2.item(1)
                        print(3)
                        calc, dd_pose = landmark_pose(pos1, im1, pos2, im2, pos, im3)
                        if calc:
                            dy_x = 0
                            dy_y = 0
                            dy_z = 0
                            for l in range(0, len(dd_pose)):
                                dy_x += abs(int(dd_pose[l][0]) - int(int(d_pose[l][0])))*0
                                dy_y += abs(int(dd_pose[l][1]) - int(int(d_pose[l][1])))*0
                                dy_z += abs(int(dd_pose[l][2]) - int(int(d_pose[l][2])))*0
                            ay = np.mean([covariance1.item(1) + covariance2.item(1) + covariance.item(1)])
                            dy_x /= (4*ay)
                            dy_y /= (4*ay)
                            dy_z /= (4*ay)
                            print('resy: ' + str([dy_x, dy_y, dy_z]))
                        else:
                            # one of the points can't estimate it
                            dy_x = 20000
                            dy_y = 20000
                            dy_z = 20000
                            print(':(')
                        # Z
                        pos1 = unknown_landmarks_data[index][2] +\
                               np.matrix([[0],
                                          [0],
                                          [1]]) * covariance1.item(2)
                        pos2 = unknown_landmarks_data[index][4][1] + \
                               np.matrix([[0],
                                          [0],
                                          [1]]) * covariance2.item(2)
                        pos3 = pos + np.matrix([[0],
                                                [0],
                                                [1]]) * covariance2.item(2)
                        print(4)
                        calc, dd_pose = landmark_pose(pos1, im1, pos2, im2, pos, im3)
                        if calc:
                            dz_x = 0
                            dz_y = 0
                            dz_z = 0
                            for l in range(0, len(dd_pose)):
                                dz_x += abs(int(dd_pose[l][0]) - int(int(d_pose[l][0])))*0
                                dz_y += abs(int(dd_pose[l][1]) - int(int(d_pose[l][1])))*0
                                dz_z += abs(int(dd_pose[l][2]) - int(int(d_pose[l][2])))*0
                            az = np.mean([covariance1.item(2) + covariance2.item(2) + covariance.item(2)])
                            dz_x /= (4*az)
                            dz_y /= (4*az)
                            dz_z /= (4*az)
                            print('resz: ' + str([dz_x, dz_y, dz_z]))
                        else:
                            # one of the points can't estimate it
                            dz_x = 20000
                            dz_y = 20000
                            dz_z = 20000
                            print(':(')
                        # position jacobian
                        gk = np.matrix([[dx_x, dx_y, dx_z],
                                        [dy_x, dy_y, dy_z],
                                        [dz_x, dz_y, dz_z]])
                        """
                        gk = np.matrix([[0, 0, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
                        for jj in d_pose:
                            # point jacobian
                            p_point = gk*robot_covariance*np.transpose(gk) + u
                            # point-rest covariance
                            if lrc_deff:
                                p_point_rest = gk*np.hstack([robot_covariance, landmark_robot_covariance.transpose()])
                                landmark_robot_covariance = np.vstack([landmark_robot_covariance,
                                                                       p_point_rest[:, [0, 1, 2]]])
                                cols, rows = p_point_rest.shape
                                landmark_l_landmark_covariance = p_point_rest[:, range(3, rows)]
                                l_landmark_landmark_covariance = landmark_l_landmark_covariance.transpose()
                                landmark_covariance = np.vstack([np.hstack([landmark_covariance,
                                                                            l_landmark_landmark_covariance]),
                                                                 np.hstack([landmark_l_landmark_covariance,
                                                                            p_point])])
                            else:
                                lrc_deff = True
                                p_point_rest = gk*robot_covariance
                                landmark_robot_covariance = p_point_rest[0:3:1]
                                landmark_covariance = p_point

                        print('robot covariance')
                        print(robot_covariance)
                        print('landmark covariance')
                        print(landmark_covariance)
                        print('landmark robot covariance')
                        print(landmark_robot_covariance)
                        print('marker_position')
                        print(marker_position)
                        print('knownlandmarks')
                        print(known_landmarks)
                        # Delete from unknown landmark
                        unknown_landmarks.pop(index)
                        unknown_landmarks_data.pop(index)
                    else:
                        # one of the points can't be found erase all
                        index = unknown_landmarks.index(i)
                        unknown_landmarks.pop(index)
                        unknown_landmarks_data.pop(index)
                        print('wrong guess')
            else:
                # Second time the landmark has been seen
                pos1 = unknown_landmarks_data[index][2]
                dist = np.sqrt((pos.item(0) - pos1.item(0))**2 +
                               (pos.item(1) - pos1.item(1))**2)
                print(dist)
                if dist >= 20:
                    unknown_landmarks_data[index][0] = True
                    if len(new) + len(old) == 1:
                        unknown_landmarks_data[index].append([newp, pos,
                                                             [covariance.item(0),
                                                              covariance.item(4),
                                                              covariance.item(8)]])
                    else:
                        inndex = new.index(i)*4
                        unknown_landmarks_data[index].append([np.array([newp[0][inndex:inndex+4]]),
                                                              pos, [covariance.item(0),
                                                                    covariance.item(4),
                                                                    covariance.item(8)]])
        else:
            # First time the landmark has been seen
            unknown_landmarks.append(i)
            if len(new) + len(old) == 1:
                print('pos' + str(pos))
                unknown_landmarks_data.append([False, newp, pos, [covariance.item(0),
                                                                  covariance.item(4),
                                                                  covariance.item(8)]])
            else:
                index = new.index(i)*4
                unknown_landmarks_data.append([False, np.array([newp[0][index:index+4]]),
                                               pos, [covariance.item(0),
                                                     covariance.item(4),
                                                     covariance.item(8)]])


matrix, rms, distortion, ret = read_camera_model()
