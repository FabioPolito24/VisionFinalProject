import traceback
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance
import math
from utils import read_all_paintings


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


# Match the features between two images using ORB and return the image showing that
def orb_features_matching(im):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im, None)
    im_db = None
    kp2 = None
    matches = None
    paintings, ids = read_all_paintings()
    for i, image in enumerate(paintings):
        print(ids[i])

        kp2, des2 = orb.detectAndCompute(image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:10]
        sum = 0
        for match in matches:
            sum += match.distance
        print(sum)
        if sum < 350:
            im_db = image
            break
        # if ids[i] == '../paintings_db/077.png':
        #     im_match = cv2.drawMatches(im, kp1, image, kp2, matches, None,
        #                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        #     cv2.imshow('matches', im_match)
        #     cv2.waitKey()

    im_match = cv2.drawMatches(im, kp1, im_db, kp2, matches[:20], None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return im_match


# Match the features between two images using AKAZE and return the image showing that
def akaze_features_matching(im, im_db):
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(im, None)
    kpts2, desc2 = akaze.detectAndCompute(im_db, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    im_match = cv2.drawMatches(im, kpts1, im_db, kpts2, matches[:20], None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    return im_match


def magic(frame):
    cv2.imshow('rect', frame)
    cv2.waitKey()
    # in the image
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    # apply bilateral filter
    gray = cv2.bilateralFilter(gray, 9, 40, 40)
    ret, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
    wall_mask = (gray > thresh1).astype(np.uint8) * 255
    dilate_kernel = np.ones((5, 5), np.uint8)
    wall_mask = cv2.dilate(wall_mask, dilate_kernel, iterations=3)
    wall_mask = cv2.erode(wall_mask, dilate_kernel, iterations=3)
    cv2.imshow('wall_mask', wall_mask)
    cv2.waitKey()
    # edged = cv2.Canny(gray, 25, 50)
    # # dilate borders
    # dilate_kernel = np.ones((5, 5), np.uint8)
    # edged = cv2.dilate(wall_mask, dilate_kernel, iterations=3)
    # im_orb = cv2.imread('../paintings_db/065.png', cv2.IMREAD_COLOR)

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img3 = frame.copy()
    rects = np.ones([len(contours), ])
    valid_contours = []

    for i, contour in enumerate(contours):
        try:
            # (x0, y0, w0, h0) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 50000:
                # cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                rects[i] = 0
                contours[i] = None
            else:
                valid_contours.append(contours[i])
                im_c = rectify(img3, contour)
                if im_c is not None:
                    cv2.imshow('rectify', im_c)
                    cv2.waitKey()

                    # ORB features matching
                    im_match = orb_features_matching(im_c)
                    plt.imshow(cv2.cvtColor(im_match, cv2.COLOR_BGR2RGB))
                    plt.show()

                    # la somma delle distanze non corrisponde necessariamente ai matching migliori
                    # probabilmente dipende anche dal numero di matching
                    # considerare un numero limitato di matching?
                    # considerare ua soglia per ogni matching per essere considerato o meno?

        except:
            traceback.print_exc()
            rects[i] = 0
            contours[i] = None

        # cv2.drawContours(img3, valid_contours, -1, (0, 255, 0), 3)



