import numpy as np
import cv2
import math
import glob
import scipy.spatial.distance
import matplotlib.pyplot as plt

NORM_FACTOR = 50


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of the smaller rectangle
    boxArea = min(boxA[2] * boxA[3], boxB[2] * boxB[3])
    # compute the intersection over union
    iou = interArea / float(boxArea)
    return iou

def print_rectangles_with_findContours(edged, frame):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = np.ones([len(contours), ])
    bounding_boxes = []
    # select only valid contours
    for i, contour in enumerate(contours):
        try:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            if w0 * h0 < 50000:
                rects[i] = 0
        except:
            rects[i] = 0
    # display only rectangles not inside other rectangles
    for i, contour in enumerate(contours):
        if rects[i]:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            for j, c in enumerate(contours):
                if rects[j]:
                    # check if the rectangles are different
                    if np.all(c == contour):
                        continue
                    else:
                        # check if c is inside contour
                        (x1, y1, w1, h1) = cv2.boundingRect(c)
                        # cv2.imshow("First", cv2.rectangle(frame.copy(), (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2))
                        # cv2.waitKey()
                        # cv2.imshow("Second", cv2.rectangle(frame.copy(), (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2))
                        # cv2.waitKey()
                        if bb_intersection_over_union((x0, y0, w0, h0), (x1, y1, w1, h1)) > 0.6:
                            if w0*h0 > w1*h1:
                                rects[j] = 0
                            else:
                                rects[i] = 0
                        # cv2.destroyAllWindows()
            if rects[i] == 1:
                cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                bounding_boxes.append([x0, y0, w0, h0])
    return frame, bounding_boxes


def isolate_painting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_frame_color = np.array([15, 19, 24])
    upper_frame_color = np.array([110, 143, 171])
    mask = cv2.inRange(hsv, lower_frame_color, upper_frame_color)
    return mask


def kmeans(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 4  # one for background and one for paintings (also one for paintings' frames?)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()
    # # disable only the cluster number 1 (turn the pixel into black)
    # masked_image = np.copy(image)
    # # convert to the shape of a vector of pixel values
    # masked_image = masked_image.reshape((-1, 3))
    # # color (i.e cluster) to disable
    # cluster = 1
    # masked_image[labels == cluster] = [0, 0, 0]
    # # convert back to original shape
    # masked_image = masked_image.reshape(image.shape)
    # # show the image
    # plt.imshow(masked_image)
    # plt.show()
    return segmented_image


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
    # Otsu thresholding
    _, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
    gray = (gray > thresh1).astype(np.uint8) * 255
    # ret = kmeans(img)

    # dilate borders
    dilate_kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(gray, dilate_kernel, iterations=2)
    return edged

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
