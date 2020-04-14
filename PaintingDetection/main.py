import cv2
from cv2 import VideoWriter_fourcc
from utils import print_rectangles_with_findContours, houghLines, isolate_painting


cap = cv2.VideoCapture('VIRB0395.MP4')

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
        edged = cv2.Canny(gray, 20, 50)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnts_img = edged.copy()
        cv2.drawContours(cnts_img, contours, -1, (255, 255, 255), 3)
        cv2.imshow('Contours', cnts_img)

        rects = print_rectangles_with_findContours(edged.copy(), frame.copy())
        cv2.imshow('Rectangles', rects)

        lines = houghLines(edged.copy())
        # lines fa talmente schifo che non vale la pena di stamparla

        hsv_isolation = isolate_painting(frame)
        cv2.imshow('HSV Isolation', hsv_isolation)

        img.append(frame)

        # cv2.imshow('Frame', edged)
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
