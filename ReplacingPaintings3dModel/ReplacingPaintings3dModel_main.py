import cv2
import matplotlib.pyplot as plt
from PaintingDetection.detection_utils import method_1, method_0, method_2

def main():
    img = cv2.imread('../screenshots_3d_model/screenshot_04.png',
                     cv2.IMREAD_COLOR)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # img = preprocessing(img)
    #
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    m1 = method_1(img)

    plt.imshow(cv2.cvtColor(m1, cv2.COLOR_BGR2RGB))
    plt.show()

    m2 = method_0(img)

    plt.imshow(cv2.cvtColor(m2, cv2.COLOR_BGR2RGB))
    plt.show()

    m3 = method_2(img)

    plt.imshow(cv2.cvtColor(m3, cv2.COLOR_BGR2RGB))
    plt.show()



    # cv2.imshow('img', img)
    # cv2.waitKey()

if __name__ == '__main__':
    main()