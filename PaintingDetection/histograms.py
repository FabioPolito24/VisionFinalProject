import cv2
import numpy as np
from PaintingDetection.general_utils import read_all_paintings


NORM_FACTOR = 50


def normalize_hist(b_hist, g_hist, r_hist):
    cv2.normalize(b_hist, b_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=NORM_FACTOR, norm_type=cv2.NORM_MINMAX)
    return b_hist, g_hist, r_hist


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


def hist_error(hist1, hist2):
    mse_b = ((hist1[0] - hist2[0]) ** 2).mean()
    mse_g = ((hist1[1] - hist2[1]) ** 2).mean()
    mse_r = ((hist1[2] - hist2[2]) ** 2).mean()
    mean = (mse_b + mse_g + mse_r) / 3
    if mean < 70:
        return True
    return False