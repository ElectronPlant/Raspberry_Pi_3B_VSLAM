import cv2
import numpy as np
__author__ = 'David Arnaiz'
"""
Basic camera navigation module: Module for basic navigation and object recognition.

-)  floor detection algorithm based on the article "http://www.hindawi.com/journals/mpe/2012/240476/"
    It's has two assumptions:
        1) Uniform floor the floor has an uniform colour all over the surface with no patterns.
        2) The objects form a visible edge with the floor so an edge detection algorithm would detect them.
    The algorithm consist in 2 functions:
        1) getFilter(image) --> filter_values: Gets the bottom part of the image and dissing a rgb colour filter for it.
        It takes an image as input and adjusts the filter for it. The function returns the values of the filter:
        (b_min, g_min, r_min, b_max, g_max, r_max).
        2) filterImage(image, filter_values) --> mask: This function applies the filter values to the image. Then
        it makes an edge detection to get a mask were th white image represents the floor and the black the rest of the
        image. The function takes two inputs the filter_values from the getFilter() function and the image itself.

-)  HSV object tracking(image, hsv_filter_values) --> mask: This function applies the filter to the image converted to
    hsv colour space and returns the mask with the object detected
"""

# Floor detection algorithm


def colourFilter(image):  # Adjusts the filter values for the filter and returns them if roi change change floorFilter()
    """
    :Input  --> image: cv2 image (480 x 640), it can be either in HSV o RGB color spaces.
    :Output --> mask: cv2 binary image, all pixels considered to be floor are white and the rest black
    This function gets the values necessary to filter the floor from the input image. To achieve this first defines the
    ROI at the button centre part of the image. Then it process the minimum and maximum colour values necessary to
    filter the image. As the colour filter is then corrected with the line detection the colour range is enlarged to
    make sore that all the floor is considered. Once the filter is designed the image is then filtered producing the
    mask were the pixels considered as floor are white and the rest is black.
    """

    # Filter design
    # ROI definition. The ROI should just include the floor
    roi = image[440:480, 150:490]  # first input y (0 at the top) then x (0 at the left), y max = 480 and x max = 640
    adj = 40  # This parameter is to increase the maximum values and decrease the minimum values found.
    # Set the parameters
    hroi = roi[:, :, 0]
    sroi = roi[:, :, 1]
    vroi = roi[:, :, 2]
    # Adjust the threshold with the parameter adj. Make sore that the minimum and maximum values are between 0 and 255
    h_min = max(0, hroi.min()-adj)
    s_min = max(0, sroi.min()-adj)
    v_min = max(0, vroi.min()-adj)
    h_max = min(255, hroi.max()+adj)
    s_max = min(255, sroi.max()+adj)
    v_max = min(255, vroi.max()+adj)

    # Apply the filter to the image
    mask = cv2.inRange(image, (int(h_min), int(s_min), int(v_min)), (int(h_max), int(s_max), int(v_max)))
    # Noise filter to fill holes in the mask image produced by shadows or lightning
    kernel = np.ones((5, 5), np.uint8)  # noise filter operator
    mask = cv2.dilate(mask, kernel, iterations=8)  # Fill holes in the area detected as floor
    mask = cv2.erode(mask, kernel, iterations=12)  # Erode de dilated mask filled holes won't be eroded
    mask = cv2.dilate(mask, kernel, iterations=4)  # Fill holes in the area detected as floor
    """
    # Just representation
    iimg = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # turn to hsv colour space
    cv2.imwrite('colourfiltermask.png', mask)
    f2 = np.zeros(image.shape, np.uint8)
    f2 = cv2.rectangle(f2, (150, 440), (490, 480), (255, 0, 0), -1)
    f3 = np.zeros(image.shape, np.uint8)
    im1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    f3 = cv2.drawContours(f3, contours, -1, (100, 0, 100), -1)
    image1 = cv2.addWeighted(iimg, 0.6, f2, 0.4, 0)
    image2 = cv2.addWeighted(iimg, 0.6, f3, 0.4, 0)
    cv2.imshow('image1', image1)
    cv2.imwrite('ROIcolourfilter.png', image1)
    cv2.imwrite('segmentedimagecolourfilter.png', image2)
    """
    return mask


def lineFilter(image, mask):  # Returns a binary image with white pixels were the floor is detected.
    """
    ------- Consider just considering the pixels connected to the main floor-----
    :Input  --> image: cv2 RGB image (480 x 640)
    :Input  --> mask: Binary image from the previous colorFilter() step
    :Output --> mask: Binary image with white filter were the floor has been detected and black in the rest
    This function completes the previous step on filtering the floor. In this case the contour of the obstacles is
    detected and all the pixels on top aren't considered as part of the floor. Then the mask is altered to make dose
    pixels black (not considered floor).
    """
    # Edge detection
    # Edge detection algorithm
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change the image to gray scale
    # edges = cv2.Canny(img, 130, 24)
    edges = cv2.Canny(gray, 200, 20, 10)  # Canny edge detection algorithm to find the edges
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change the image to gray scale
    # Noise filtering
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur to take the noise out
    # Find contours
    edges = cv2.Canny(gray, 200, 20, 3)  # Canny edge detection algorithm to find the edges
    # Hough probabilistic transform to the edges found
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=40, maxLineGap=20)
    try:  # Prevents crash when no line in detected (type = NoneType)
        if lines is not None:
            """
            for i in lines[:]:
                x1, y1, x2, y2 = i[0, :]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            """
        # Change the mask
            for i in range(len(lines)):
                for (x1, y1, x2, y2) in lines[i]:
                    pts = np.array([[x1, y1], [x2, y2], [x2, 0], [x1, 0]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    mask = cv2.fillConvexPoly(mask, pts, 0)
    except (RuntimeError, TypeError, NameError):
        pass
    # Another noise reduction filter to erase small dots of noise in either black or white
    kernel = np.ones((5, 5), np.uint8)  # noise filter operator
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Erase small white dots from the image
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Erase small black dots from the image
    """
    # just representation
    cv2.imwrite('masklinefilter3.png', mask)
    f3 = np.zeros(image.shape, np.uint8)
    im1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    f3 = cv2.drawContours(f3, contours, -1, (100, 0, 100), -1)
    image2 = cv2.addWeighted(image, 0.6, f3, 0.4, 0)
    cv2.imwrite('edgeslinefilter3.png', edges)
    cv2.imwrite('lineslinefilter3.png', image)
    cv2.imwrite('mask2linefilter3.png', image2)
    """
    return mask


def floorFilter(frame):  # Filter the parts of the image that are floor
    """
    :Input  --> frame: cv2 RGB image (480 x 640)
    :Output --> mask: Binary image with white filter were the floor has been detected and black in the rest
    This function calls the colourFilter() and the lineFilter() functions. This function classifies the pixels that are
    floor from the ones that aren't. The method is first filter the floor by it HSV colour values, and then detects the
    contours of the obstacles with a line detection algorithm and considers the pixels on top of the contour to be
    obstacles.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # turn to hsv colour space
    mask = colourFilter(hsv)
    mask = lineFilter(frame, mask)
    im1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Only consider as floor the pixels connected in the mask with the ROI
    try:  # Prevents crash when no contour in detected (type = NoneType)
        contours2 = contours
        k = 0
        mask2 = np.zeros(frame.shape, np.uint8)
        mask2 = cv2.rectangle(mask2, (150, 420), (490, 480), 255, -1)  # Draw the ROI on the second mask
        for i in range(len(contours)):
            # Approximate the contour reducing its vertices
            epsilon = 0.01*cv2.arcLength(contours[i], False)  # Maximum error is 10% of the contours perimeter
            approx = cv2.approxPolyDP(contours[i], epsilon, False)  # edges reduction
            area = cv2.contourArea(approx)
            if area >= 600:  # To reduce the contours analysed
                # check if its in contact with the ROI
                mask3 = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask3, [approx], 0, 255, -1)
                mask3 *= mask2
                cv2.imshow('mask3', mask3)
                pixelpoints = np.transpose(np.nonzero(mask3))
                if len(pixelpoints) == 0:
                    contours2 = np.delete(contours2, k)
                else:
                    #  contours2[k] = approx
                    k += 1
            else:
                contours2 = np.delete(contours2, k)
        if len(contours2) > 0:
            # showing only
            mmask = np.zeros(frame.shape, np.uint8)
            mmask = cv2.drawContours(mmask, contours2, -1, (255, 255, 255), -1)
            cv2.imwrite('maskfloorfilter.png', mmask)
            cv2.imwrite('mmmmmask.png', mask)
            f2 = np.zeros(frame.shape, np.uint8)
            cv2.drawContours(f2, contours2, -1, (255, 0, 255), -1)
            f3 = np.zeros(frame.shape, np.uint8)
            cv2.rectangle(f3, (23, 375), (480, 540), (0, 100, 100), -1)
            cv2.rectangle(f3, (10, 375), (63, 480), (0, 255, 255), -1)
            cv2.rectangle(f3, (440, 375), (600, 480), (0, 255, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, f2, 0.3, 0)
            frame = cv2.addWeighted(frame, 0.7, f3, 0.3, 0)
            # Check for objects
            mask *= 0
            cv2.drawContours(mask, contours2, -1, 255, -1)
            mask = cv2.bitwise_not(mask)
            roi = mask[375:479, 23:540]  # chop image
            cv2.imwrite('roi.png', mask)
            pixelpoints = np.transpose(np.nonzero(roi))
            if not len(pixelpoints) == 0:  # objects detected in the roi
                # Check in what side is the object in
                lroi = mask[375:479, 10:63]  # left side
                rroi = mask[375:479, 500:630]  # right side
                cv2.imshow('d', rroi)
                cv2.imshow('i', lroi)
                lpixelpoints = np.transpose(np.nonzero(lroi))
                rpixelpoints = np.transpose(np.nonzero(rroi))
                if not len(lpixelpoints) == 0 and not len(rpixelpoints) == 0:  # object in both sides
                    # turn right
                    cv2.imwrite('both_edited_frame.png', frame)
                    return True, 'both'
                elif not len(lpixelpoints) == 0:  # object on  the left
                    # turn right
                    cv2.imwrite('left_edited_frame.png', frame)
                    return True, 'left'
                elif not len(rpixelpoints) == 0:  # object on the right
                    # turn left
                    cv2.imwrite('right_edited_frame.png', frame)
                    return True, 'right'
                else:  # small object in the middle
                    # turn right
                    cv2.imwrite('middle_edited_frame.png', frame)
                    return True, 'middle'
            else:
                return False, 'no'
    except (RuntimeError, TypeError, NameError):
        pass
    cv2.imshow('frame', frame)
    return False, 'no'
