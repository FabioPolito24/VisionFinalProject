import numpy as np
import cv2
import math
import glob
import scipy.spatial.distance

NORM_FACTOR = 50


def print_rectangles_with_findContours(edged, frame, b_hist, g_hist, r_hist):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = np.ones([len(contours), ])
    bounding_boxes = []
    # seleziono solo i contorni validi
    for i, contour in enumerate(contours):
        try:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            if w0 * h0 < 50000:
                # cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                rects[i] = 0
        except:
            rects[i] = 0
    # stampo i rettangoli che non sono contenuti dentro altri rettangoli
    for i, contour in enumerate(contours):
        if rects[i]:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            for j, c in enumerate(contours):
                if rects[j]:
                    # se contour e c sono lo stesso contorno, vado avanti
                    if np.all(c == contour):
                        continue
                    else:
                        # controllo se contour è contenuto in c
                        (x1, y1, w1, h1) = cv2.boundingRect(c)
                        if x1 > x0 and y1 > y0 and x1 + w1 < x0 + w0 and y1 + h1 < y0 + h0:
                            # contour è contenuto in c quindi lo tolgo dalla lista
                            rects[i] = 0
                            break
            if rects[i] == 1:
                # seleziono solo la parte interna del rettangolo così evito l'eventuale cornice che nel database non è quasi mai presente
                y_range = [round(y0 + h0 / 5), round(y0 + h0 * 4 / 5)]
                x_range = [round(x0 + w0 / 5), round(x0 + w0 * 4 / 5)]
                img_for_hist = frame[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
                # cv2.imshow('rect', img_for_hist)
                # cv2.waitKey(0)
                b, g, r = get_hist(img_for_hist)
                if hist_error([b, g, r], [b_hist, g_hist, r_hist]):
                    cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                    bounding_boxes.append([x0, y0, w0, h0])
    return frame, bounding_boxes


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


def isolate_painting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_frame_color = np.array([15, 19, 24])
    upper_frame_color = np.array([110, 143, 171])
    mask = cv2.inRange(hsv, lower_frame_color, upper_frame_color)
    return mask


def preprocessing(frame):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply bilateral filter
    gray = cv2.bilateralFilter(gray, 9, 40, 40)
    return gray


def method_0(frame):
    gray = preprocessing(frame)
    # Canny edge detection
    edged = cv2.Canny(gray, 25, 50)
    # cv2.imshow('Canny', edged)

    # dilate borders
    dilate_kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, dilate_kernel, iterations=2)
    # cv2.imshow('Canny + dilate', edged)
    return edged


def method_1(frame):
    gray = preprocessing(frame)
    # #Otsu thresholding
    _, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
    gray = (gray > thresh1).astype(np.uint8) * 255

    # dilate borders
    dilate_kernel = np.ones((8, 8), np.uint8)
    edged = cv2.dilate(gray, dilate_kernel, iterations=2)
    return edged


def read_all_paintings():
    images = glob.glob("../paintings_db/*.png")
    paintings = []
    for image in images:
        img = cv2.imread(image)
        paintings.append(img)
    # for i, img in enumerate(paintings):
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)
    return paintings


def get_mean_hist():
    imgs = read_all_paintings()
    histSize = 256
    histRange = (0, 256)  # the upper boundary is exclusive
    b_hist = np.zeros((len(imgs), 256, 1))
    g_hist = np.zeros((len(imgs), 256, 1))
    r_hist = np.zeros((len(imgs), 256, 1))
    accumulate = False
    for i, src in enumerate(imgs):
        bgr_planes = cv2.split(src)
        b_hist[i] = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist[i] = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist[i] = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    # bin_w = int(np.round(hist_w / histSize))
    # histImage = np.zeros((NORM_FACTOR, hist_w, 3), dtype=np.uint8)
    b_hist = np.mean(b_hist, axis=0)
    g_hist = np.mean(g_hist, axis=0)
    r_hist = np.mean(r_hist, axis=0)
    # b_hist, g_hist, r_hist = normalize_hist(b_hist, g_hist, r_hist)
    # for i in range(1, histSize):
    #     cv2.line(histImage, (bin_w * (i - 1), NORM_FACTOR - int(np.round(b_hist[i - 1]))),
    #             (bin_w * i, NORM_FACTOR - int(np.round(b_hist[i]))),
    #             (255, 0, 0), thickness=2)
    #     cv2.line(histImage, (bin_w * (i - 1), NORM_FACTOR - int(np.round(g_hist[i - 1]))),
    #             (bin_w * i, NORM_FACTOR - int(np.round(g_hist[i]))),
    #             (0, 255, 0), thickness=2)
    #     cv2.line(histImage, (bin_w * (i - 1), NORM_FACTOR - int(np.round(r_hist[i - 1]))),
    #             (bin_w * i, NORM_FACTOR - int(np.round(r_hist[i]))),
    #             (0, 0, 255), thickness=2)
    # cv2.imshow('calcHist Demo', histImage)
    # cv2.waitKey()
    return normalize_hist(b_hist, g_hist, r_hist)


def get_hist(src):
    if src is None:
        print('Could not open or find the image')
        exit(0)
    bgr_planes = cv2.split(src)
    histSize = 256
    histRange = (0, 256)  # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    return normalize_hist(b_hist, g_hist, r_hist)


def hist_error(hist1, hist2):
    mse_b = ((hist1[0] - hist2[0]) ** 2).mean()
    mse_g = ((hist1[1] - hist2[1]) ** 2).mean()
    mse_r = ((hist1[2] - hist2[2]) ** 2).mean()
    mean = (mse_b + mse_g + mse_r) / 3
    if mean < 70:
        return True
    return False


def normalize_hist(b_hist, g_hist, r_hist):
    cv2.normalize(b_hist, b_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    return b_hist, g_hist, r_hist


get_mean_hist()

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
def orb_features_matching(im, im_db):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im, None)
    kp2, des2 = orb.detectAndCompute(im_db, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
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
