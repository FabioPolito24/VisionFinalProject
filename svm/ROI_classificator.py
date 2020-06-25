import _pickle as pickle
import cv2
import numpy as np
import sklearn as sk
import numpy as np
from sklearn import svm


class clf_svm:
    class __clf_svm:
        def __init__(self):
            with open('../svm/model.pickle', 'rb') as model_file:

                self.clf = pickle.load(model_file)

    instance = None

    def __init__(self):
        if not clf_svm.instance:
            clf_svm.instance = clf_svm.__clf_svm()

    def get_clf(self):
        return self.instance.clf


def create_hist(roi):
    bgr_planes = cv2.split(roi)
    histSize = 256
    histRange = (0, 256)  # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    b_hist *= 100 / b_hist.sum()
    g_hist *= 100 / b_hist.sum()
    r_hist *= 100 / b_hist.sum()
    return np.concatenate((b_hist, g_hist, r_hist)).transpose();

def check_roi(roi):
    clf = clf_svm().get_clf()
    h = create_hist(roi)
    return clf.predict(h)
