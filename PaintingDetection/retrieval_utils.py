import cv2
import numpy as np

from general_utils import read_all_paintings


DEBUG = False


# Match the features between two images using ORB and return the image showing that
def orb_features_matching(im):
    orb = cv2.ORB_create()
    images, filenames = read_all_paintings()
    top_5_im = [{'im': None, 'filename': None}] * 5
    top_5_score = np.zeros((5,))
    if DEBUG:
        # # add brightness to the image
        # new_image = np.zeros(im.shape, im.dtype)
        # for y in range(im.shape[0]):
        #     for x in range(im.shape[1]):
        #         for c in range(im.shape[2]):
        #             new_image[y, x, c] = np.clip(1.3 * im[y, x, c] + 40, 0, 255)
        # cv2.imshow('Original Image', im)
        # cv2.imshow('New Image', new_image)
        # # Wait until user press some key
        # cv2.waitKey()
        new_image = im.copy()
    else:
        new_image = im.copy()
    kp1, des1 = orb.detectAndCompute(new_image, None)
    for i, im_db in enumerate(images):
        kp2, des2 = orb.detectAndCompute(im_db, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        im_match = cv2.drawMatches(new_image, kp1, im_db, kp2, matches[:20], None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        if DEBUG:
            cv2.imshow("matches", im_match)
            cv2.waitKey()
        # score created as a weighted sum of matches and distances
        # better the matches, smaller the average distance, so we subtract this average to a symbolic number (150)
        score = len(matches) * 0.3 + 150 - np.mean([data.distance for data in matches[:10]]) * 0.7
        if score > top_5_score[-1]:
            top_5_score[-1] = score
            top_5_score[::-1].sort()
            j, = np.where(np.isclose(top_5_score, score))
            if len(j) == 1:
                top_5_im[j[0]]['im'] = im_db
                top_5_im[j[0]]['filename'] = filenames[i]
            else:
                print("[ORB_MATCHING]: 2 images from database with the same score")
                top_5_im[j[0]] = im_db
    return top_5_im


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
