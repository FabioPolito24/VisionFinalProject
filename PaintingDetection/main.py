import cv2
import numpy as np
from cv2 import VideoWriter_fourcc
from utils import print_rectangles_with_findContours, houghLines, isolate_painting
from tkinter import *
from tkinter.ttk import *


def analyze():
    cap = cv2.VideoCapture(entry.get())

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

            # Applichiamo un filtro
            # gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
            gray = cv2.bilateralFilter(gray, 9, 40, 40)

            # #Otsu thresholding
            # _, thresh1 = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
            # gray = (gray > thresh1).astype(np.uint8) * 255
            # # cv2.imshow('Otsu Threshold', gray)

            # Canny edge detection
            edged = cv2.Canny(gray, 25, 50)
            # cv2.imshow('Canny', edged)

            # Dilata/erodi i bordi ottenuti con Canny
            dilate_kernel = np.ones((5, 5), np.uint8)
            edged = cv2.dilate(edged, dilate_kernel, iterations=2)
            cv2.imshow('Canny + dilate', edged)
            # edged = cv2.erode(edged, dilate_kernel, iterations=2)

            # #Cerca i contorni
            # contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #
            # #Disegna i Contorni trovati in verde
            # cnts_img = edged.copy()
            # cv2.drawContours(cnts_img, contours, -1, (255, 255, 255), 3)
            # cv2.imshow('Contours', cnts_img)

            rects, bounding_boxes = print_rectangles_with_findContours(edged.copy(), frame.copy())
            cv2.imshow('Rectangles', rects)

            # lines = houghLines(edged.copy())
            # lines fa talmente schifo che non vale la pena di stamparla

            # hsv_isolation = isolate_painting(frame)
            # cv2.imshow('HSV Isolation', hsv_isolation)

            img.append(rects)

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


tk = Tk()
tk.geometry("300x60")
tk.title("Video Analyzer")
label = Label(tk, text="Insert path to a video").pack()
entry = Entry()
entry.insert(0, "../videos/vid001.MP4")
entry.pack()
submit = Button(tk, text="Submit", command=analyze).pack()

tk.mainloop()
