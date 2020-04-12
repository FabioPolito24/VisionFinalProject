import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from random import randrange


def main():

    img = cv.imread('../example_imgs/gallery_1.jpg',cv.IMREAD_COLOR)
    '''
    #img Shape = (H,W,3)
    cv.imshow("Source", img)
    #cv.waitKey()

    #img = cv.bilateralFilter(img,9,40,40)
    #cv.imshow("Bilateral", img)

    #img = cv.pyrMeanShiftFiltering(img, sp = 1, sr = 25,maxLevel=1)
    #cv.imshow("mean shift", img)

    #converting img to black and white img
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow("grey img", gray_img)

    #Option 1: Adaptive thresholding
    '''
    # wall_mask = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,31, 10)
    # cv.imshow("adaptive_thresh", wall_mask)
    # dilate_kernel = np.ones((3, 3), np.uint8)
    # wall_mask = cv.dilate(wall_mask, dilate_kernel, iterations=1)
    # cv.imshow("Dilating", wall_mask)
    '''

    # Option 2: Global thresholding otsu
    ret, thresh1 = cv.threshold(gray_img, 120, 255, cv.THRESH_OTSU)
    wall_mask = (gray_img > thresh1).astype(np.uint8) * 255
    #cv.imshow("Otsu thresholds", wall_mask)
    erode_kernel = np.ones((4, 4), np.uint8)
    dilate_kernel = np.ones((4, 4), np.uint8)
    wall_mask = cv.dilate(wall_mask, dilate_kernel, iterations=2)
    #cv.imshow("Dilate1", wall_mask)
    wall_mask = cv.erode(wall_mask, erode_kernel, iterations=10)
    #cv.imshow("Erode", wall_mask)
    wall_mask = cv.dilate(wall_mask, dilate_kernel, iterations=8)
    #cv.imshow("Dilate2", wall_mask)

    #Overlap mask with original image
    rgb_wall_mask = cv.cvtColor(wall_mask, cv.COLOR_GRAY2BGR)
    overlap_img = cv.bitwise_and(img, rgb_wall_mask)
    #cv.imshow("Overlapped img", overlap_img)

    # Edge detection
    edges = cv.Canny(wall_mask, 50, 100)
    cv.imshow("Edges with Canny", edges)

    #Dilate before pass to hough
    dilate_kernel = np.ones((4, 4), np.uint8)
    edges = cv.dilate(edges, dilate_kernel, iterations=1)


    # Copy edges to the images that will display the results in BGR
    cimg = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    threshold = 80
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, threshold, 0, 0)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cimg, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Hough Line Transform", cimg)
    cv.waitKey()
    '''
    #-------------------Method found on github ACTUALLY NOT WORKING----------------------

    cl = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    color_difference = np.full((3,1), 2, dtype=np.uint8)
    wall_color = np.random.randint(256, size=3, dtype=np.uint8)
    largest_segment = 0

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y+1, x+1] == 0:
                new_val = tuple(np.random.randint(255, size=3, dtype=np.uint8))
                retval1,ret2,ret3, rect = cv.floodFill(cl, mask, (x,y), (255,0,0), flags=cv.FLOODFILL_FIXED_RANGE)  # ho sostituito tra i parametri della funzione new_val con (255, 0, 0) per farlo compilare
                segment_size = len(rect)
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wall_color = new_val
    wall_color1 = np.zeros((3,1), dtype=np.uint8)
    wall_color2 = np.full((3,1), 100, dtype=np.uint8)
    wall_mask = cv.inRange(img, wall_color1, wall_color2)
    cv.imshow("Wall mask", wall_mask)

    rgb_wall_mask = cv.cvtColor(wall_mask, cv.COLOR_GRAY2BGR)
    overlap_img = cv.bitwise_and(img, rgb_wall_mask)
    cv.imshow("Overlapped img", overlap_img)

    cv.waitKey()


if __name__ == '__main__':
    main()