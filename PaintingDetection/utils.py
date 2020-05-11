import numpy as np
import cv2
import math
import glob

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
