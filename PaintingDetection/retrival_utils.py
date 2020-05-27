from detection_utils import read_all_paintings
from rectification_utils import rectify
import numpy as np
import matplotlib.pyplot as plt
import cv2
import traceback


# Match the features between two images using ORB and return the image showing that
def orb_features_matching(im, im_db):
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