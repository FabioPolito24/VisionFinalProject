import cv2
import numpy as np
from PaintingDetection.general_utils import read_all_paintings
import _pickle as pickle
import time

DEBUG = False


class PaintingsDB:
    class __PaintingsDB:
        def __init__(self):
            with open('../paintings_db/db_paintings.pickle', 'rb') as db_paintings_file:

                self.paintings = pickle.load(db_paintings_file)

                # After config_dictionary is read from file
                for i, painting in enumerate(self.paintings):
                    for k, point in enumerate(painting['kp']):
                        self.paintings[i]['kp'][k] = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                                                  _angle=point[2], _response=point[3],
                                                                  _octave=point[4], _class_id=point[5])
    instance = None

    def __init__(self):
        if not PaintingsDB.instance:
            PaintingsDB.instance = PaintingsDB.__PaintingsDB()

    def get_db(self):
        return self.instance.paintings


# Load the DB's paintings
def load_db_paintings():
    with open('../paintings_db/db_paintings.pickle', 'rb') as db_paintings_file:

        paintings = pickle.load(db_paintings_file)

        # After config_dictionary is read from file
        for i, painting in enumerate(paintings):
            for k, point in enumerate(painting['kp']):
                paintings[i]['kp'][k] = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                                                     _response=point[3], _octave=point[4], _class_id=point[5])

    return paintings


# Use ORB to find the top 5 matches between im and the DB's paintings.
# Return a list of dictionaries containing the top 5 matched paintings, their name and the numbers of matches
def orb_features_matching(im):
    db_paintings = PaintingsDB().get_db()
    orb = cv2.ORB_create()
    top_5_im = [{'im': None, 'filename': None, 'score': None}] * 5
    top_5_score = np.full((5,), -1)
    total = 0
    kp1, des1 = orb.detectAndCompute(im, None)
    for painting in db_paintings:
        # crossCheck=True alternative to D.Lowe method
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, painting['des'], k=2)
        # Apply ratio test from D.Lowe in sift paper
        good = []
        for m, n in matches:
            # as the hyperparameter get closer to 1, more key points will be matched
            if m.distance < 0.70 * n.distance:
                good.append([m, n])
        # im_match = cv2.drawMatchesKnn(im, kp1, painting['im'], painting['kp'], good, None, flags=2)
        # cv2.imshow("matches", im_match)
        # cv2.waitKey()
        # total += len(good);
        if len(good) > top_5_score.min():
            top_5_im[top_5_score.argmin()] = {'im': painting['im'], 'filename': painting['filename'], 'score': len(good)}
            top_5_score[top_5_score.argmin()] = len(good)

    total_top_5 = 0
    # Sort the best 5 matches found
    top_5_match = sorted(top_5_im, key=lambda k: k['score'], reverse=True)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Some useful information to understand the result of the function
    # cv2.imshow("Original", im)
    # for i, score in enumerate(top_5_match):
    #     print("match number " + str(i) + " with score " + str(top_5_match[i]['score']))
    #     cv2.imshow(top_5_match[i]['filename'] + " number " + str(i), top_5_match[i]['im'])
    #     total_top_5 += top_5_match[i]['score']

    # Some metric that could help understand if the painting has been found in the DB
    # print("Total score =   " + str(total))
    # print("mean score =   " + str(total / 95))
    # print("mean top 5 = " + str(total_top_5 / 5))
    # cv2.waitKey()

    return top_5_match

# Use flann matcher but doesn't work, sometimes the knnMatch don't return 2 value and I don't know why
def orb_features_matching_flann(im):
    db_paintings = PaintingsDB().get_db()
    orb = cv2.ORB_create()
    top_5_im = [{'im': None, 'filename': None, 'score': None}] * 5
    top_5_score = np.full((5,), -1)
    total = 0
    # start_time = time.time()
    kp1, des1 = orb.detectAndCompute(im, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=30)  # or pass empty dictionary

    for painting in db_paintings:
        # crossCheck=True alternative to D.Lowe method
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        fl = cv2.FlannBasedMatcher(index_params,search_params)
        matches = fl.knnMatch(des1, painting['des'], k=2)
        # Apply ratio test from D.Lowe in sift paper
        good = []
        for m, n in matches:
            # as the hyperparameter get closer to 1, more key points will be matched
            if m.distance < 0.70 * n.distance:
                good.append([m, n])
        # im_match = cv2.drawMatchesKnn(im, kp1, painting['im'], painting['kp'], good, None, flags=2)
        # cv2.imshow("matches", im_match)
        # cv2.waitKey()
        total += len(good);
        if len(good) > top_5_score.min():
            top_5_im[top_5_score.argmin()] = {'im': painting['im'], 'filename': painting['filename'], 'score': len(good)}
            top_5_score[top_5_score.argmin()] = len(good)

    # total_top_5 = 0
    # Sort the best 5 matches found
    top_5_match = sorted(top_5_im, key=lambda k: k['score'], reverse=True)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Some useful information to understand the result of the function
    # cv2.imshow("Original", im)
    # for i, score in enumerate(top_5_match):
    #     print("match number " + str(i) + " with score " + str(top_5_match[i]['score']))
    #     cv2.imshow(top_5_match[i]['filename'] + " number " + str(i), top_5_match[i]['im'])
    #     total_top_5 += top_5_match[i]['score']

    # Some metric that could help understand if the painting has been found in the DB
    # print("Total score =   " + str(total))
    # print("mean score =   " + str(total / 95))
    # print("mean top 5 = " + str(total_top_5 / 5))
    # cv2.waitKey()

    return top_5_match


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

