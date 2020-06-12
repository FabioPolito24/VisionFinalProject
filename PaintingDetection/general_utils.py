import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def kmeans(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3  # one for background and one for paintings (also one for paintings' frames?)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()
    # # disable only the cluster number 1 (turn the pixel into black)
    # masked_image = np.copy(image)
    # # convert to the shape of a vector of pixel values
    # masked_image = masked_image.reshape((-1, 3))
    # # color (i.e cluster) to disable
    # cluster = 1
    # masked_image[labels == cluster] = [0, 0, 0]
    # # convert back to original shape
    # masked_image = masked_image.reshape(image.shape)
    # # show the image
    # plt.imshow(masked_image)
    # plt.show()
    return segmented_image


def watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # plt.imshow(sure_bg, cmap='gray')
    # plt.show()
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    # plt.imshow(sure_fg, cmap='gray')
    # plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow('watershed', img)
    cv2.waitKey()
    return img

