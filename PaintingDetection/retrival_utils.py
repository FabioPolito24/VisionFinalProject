import numpy as np
import cv2
import math
import glob
import scipy.spatial.distance


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
