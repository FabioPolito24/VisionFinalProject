from PaintingDetection.detection_utils import *
from PaintingDetection.rectification_utils import alignImages
from svm.ROI_classificator import check_roi

DELTA = 10


def main_3d(img):
    orig = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = preprocessing(hsv)
    # Otsu thresholding
    _, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
    gray = (gray < thresh1).astype(np.uint8) * 255
    erode_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)
    m1 = cv2.dilate(gray, dilate_kernel, iterations=5)
    m1 = cv2.erode(m1, erode_kernel, iterations=5)
    m2 = cv2.erode(gray, erode_kernel, iterations=5)
    m2 = cv2.dilate(m2, dilate_kernel, iterations=5)
    cnts1 = cv2.findContours(m1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)
    cnts1 = sorted(cnts1, key=cv2.contourArea, reverse=True)[:15]
    cnts2 = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:15]
    c1 = 0
    c2 = 0
    for c in cnts1:
        if cv2.contourArea(c) < 100:
            continue
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        (x0, y0, w0, h0) = cv2.boundingRect(c)
        # if len(approx) == 4:
        if check_roi(img[y0:y0 + h0, x0:x0 + w0]):
            c1 += 1
    for c in cnts2:
        if cv2.contourArea(c) < 100:
            continue
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        (x0, y0, w0, h0) = cv2.boundingRect(c)
        # if len(approx) == 4:
        if check_roi(img[y0:y0 + h0, x0:x0 + w0]):
            c2 += 1
    if c1 >= c2:
        cv2.imshow('b&w', cv2.resize(m1, (700, 500)))
        cnts = cnts1
    else:
        cv2.imshow('b&w', cv2.resize(m2, (700, 500)))
        cnts = cnts2
    cv2.waitKey()

    black_img = np.zeros_like(img)
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            canvas = black_img.copy()
            cv2.drawContours(canvas, [approx], -1, (255, 255, 255), -1)
            mask = (canvas == 255).all(axis=2)
            black_img[mask] = img[mask]
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.imshow('cnt', cv2.resize(black_img, (700, 500)))
            # cv2.imshow('bb', img[y:y + h, x:x + w, :])
            cv2.waitKey()
            top_5_matches, top_5_scores = orb_features_matching(img[y:y + h, x:x + w, :])
            if top_5_scores[0] - np.mean(top_5_scores) > np.ceil(np.mean(top_5_scores) / 3):
                cv2.imshow('match 0', top_5_matches[0]['im'])
                # cv2.imshow('match 1', top_5_matches[1]['im'])
                sub_image = img[y:y + h, x:x + w, :]
                cv2.imshow('sub_image', sub_image)
                aligned = alignImages(top_5_matches[0]['im'], sub_image)
                cv2.imshow('aligned -1', aligned)
                # cv2.imshow('mask_pre', cv2.resize((mask < 1).astype(np.uint8) * 255, (700, 500)))
                try:
                    mask = mask[y:y + h, x:x + w]
                    sub_image[mask] = aligned[mask]
                except:
                    pass
                cv2.imshow('new', sub_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.imshow('before', cv2.resize(img, (800, 650)))
    img[img == (0,0,0)] = orig[img == (0,0,0)]
    img = cv2.resize(img, (1200, 1000))
    cv2.imshow('after', cv2.resize(img, (800, 650)))
    cv2.waitKey()
    return img


# BEFORE RUNNING THIS FILE, change line 11 of retrieval_utils.py:
# from
#    with open('paintings_db/db_paintings.pickle', 'rb') as db_paintings_file:
# to
#    with open('../paintings_db/db_paintings.pickle', 'rb') as db_paintings_file:

# BEFORE RUNNING THIS FILE, change line 11 of ROI_classificator.py:
# from
#       with open('svm/model.pickle', 'rb') as model_file:
# to
#       with open('../svm/model.pickle', 'rb') as model_file:
if __name__ == '__main__':
    img = cv2.imread('../screenshots_3d_model/screenshot_03.png', cv2.IMREAD_COLOR)
    main_3d(img)
