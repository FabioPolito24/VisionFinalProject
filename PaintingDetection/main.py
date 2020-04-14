import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
import math


cap = cv2.VideoCapture('VIRB0398.MP4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

img = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        edged = cv2.Canny(gray, 40, 100)

        lines = cv2.HoughLines(edged, 1, np.pi / 180, 130, None, 0, 0)
        # Draw the lines
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(edged, pt1, pt2, (255, 255, 255), 3)

        img.append(edged)

        cv2.imshow('Frame', edged)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

try:
    height, width, layers = img[1].shape
    fourcc = VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('video.avi', fourcc, 24, (width, height))
    for j in range(len(img)):
        video.write(img[j])
    video.release()
except:
    print('Video build failed')

cap.release()
cv2.destroyAllWindows()
