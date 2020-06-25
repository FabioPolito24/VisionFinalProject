import cv2
import matplotlib.pyplot as plt
import numpy as np

from PaintingDetection.detection_utils import *
from PaintingDetection.rectification_utils import alignImages


def main():
    img = cv2.imread('../screenshots_3d_model/screenshot_01.jpg',
                     cv2.IMREAD_COLOR)

    m2 = method_2(img)
    cv2.imshow('m2', cv2.resize(m2, (700, 550)))
    cv2.waitKey()
    orig = img.copy()
    contours, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = np.ones([len(contours), ])
    bounding_boxes = []
    try:
        # select only valid contours
        for i, contour in enumerate(contours):
            try:
                (x0, y0, w0, h0) = cv2.boundingRect(contour)
                if w0 * h0 < 40000:
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
                            if bb_intersection_over_union((x0, y0, w0, h0), (x1, y1, w1, h1)) > 0.6:
                                if w0 * h0 > w1 * h1:
                                    rects[j] = 0
                                else:
                                    rects[i] = 0
                if rects[i] == 1:  # and check_roi(img[y0:y0 + h0, x0:x0 + w0]):
                    cv2.imshow('rect', img[y0:y0 + h0, x0:x0 + w0, :])
                    cv2.waitKey()
                    cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 4)
                    bounding_boxes.append((x0, y0, w0, h0))
                    ret, top5_matches, aligned_img, mask = second_step(orig[
                                                                 max(0, y0 - DELTA): min(orig.shape[0], y0 + h0 + DELTA),
                                                                 max(0, x0 - DELTA): min(orig.shape[1], x0 + w0 + DELTA),
                                                                 :])
                    cv2.imshow('match 0', top5_matches[0]['im'])
                    cv2.imshow('match 1', top5_matches[1]['im'])
                    # cv2.imshow('match 2', top5_matches[2]['im'])
                    sub_image = orig[max(0, y0 - DELTA): min(orig.shape[0], y0 + h0 + DELTA),
                                     max(0, x0 - DELTA): min(orig.shape[1], x0 + w0 + DELTA), :]
                    aligned = alignImages(top5_matches[0]['im'], sub_image)
                    cv2.imshow('aligned -1', aligned)
                    try:
                        sub_image[mask] = aligned[mask]
                    except:
                        pass
                    cv2.imshow('new', sub_image)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
    except:
        pass
    cv2.imshow('img', cv2.resize(img, (700, 550)))
    cv2.waitKey()
    orig = cv2.resize(orig, (1200, 1000))
    return orig


if __name__ == '__main__':
    img = main()
    cv2.imshow('img', img)
    cv2.waitKey()