import glob
import cv2


def read_all_paintings():
    names = glob.glob("../paintings_db/*.png")
    paintings = []
    filenames = []
    for name in names:
        img = cv2.imread(name)
        paintings.append(img)
        filenames.append(name)
    # for i, img in enumerate(paintings):
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)
    return paintings, filenames
