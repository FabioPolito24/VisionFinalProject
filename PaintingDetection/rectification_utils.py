import numpy as np
import cv2
import math
import glob
import scipy.spatial.distance

def houghLines(edged):
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 50, None, 0, 0)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(edged, pt1, pt2, (255, 255, 255), 3)
    return edged


# Reorder the points in the correct way
def reorder(points):
    ordered_points = np.zeros((4, 2), np.int32)
    add = points.sum(1)
    ordered_points[0] = points[np.argmin(add)]
    ordered_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[2] = points[np.argmax(diff)]
    return ordered_points


# Straighten the painting given his contour, the contour has to be a quadrilateral
def rectify(frame, contour):
    # getting the 4 vertices
    epsilon = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # contours print on the frame
    # cv2.drawContours(frame, approx, -1, (0, 0, 255), 50)
    # cv2.drawContours(frame, contour, -1, (0, 255, 0), 5)

    if len(approx) == 4:
        (rows, cols, _) = frame.shape

        # image center
        u0 = cols / 2.0
        v0 = rows / 2.0

        p = reorder(approx.reshape((4, 2)))

        # widths and heights of the projected image
        # if one of the following value is zero it throws an error, but if this happens
        # it means that the shape that the algorithm has found is not a square
        w1 = scipy.spatial.distance.euclidean(p[0], p[1])
        w2 = scipy.spatial.distance.euclidean(p[2], p[3])

        h1 = scipy.spatial.distance.euclidean(p[0], p[2])
        h2 = scipy.spatial.distance.euclidean(p[1], p[3])

        # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        # plt.show()

        w = max(w1, w2)
        h = max(h1, h2)

        # visible aspect ratio
        ar_vis = float(w) / float(h)

        # make numpy arrays and append 1 for linear algebra
        m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
        m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
        m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
        m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

        # calculate the focal disrance
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1

        n21 = n2[0]
        n22 = n2[1]
        n23 = n2[2]

        n31 = n3[0]
        n32 = n3[1]
        n33 = n3[2]

        f = math.sqrt(np.abs((1.0 / (n23 * n33)) * (
                (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
                n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        # calculate the real aspect ratio
        ar_real = math.sqrt(
            np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = np.array(p).astype('float32')
        pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        # project the image with the new w/h
        # M = cv2.getPerspectiveTransform(pts1, pts2)
        #
        # im_c = cv2.warpPerspective(frame, M, (W, H))

        h, _ = cv2.findHomography(pts2, pts1)
        im_c = cv2.warpPerspective(frame, h, (W, H), flags=cv2.WARP_INVERSE_MAP)

        return im_c