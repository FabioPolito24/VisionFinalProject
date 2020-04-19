import cv2
import numpy as np
from cv2 import VideoWriter_fourcc
from utils import print_rectangles_with_findContours
from tkinter import *
from PIL import Image, ImageTk


def analyze():
    cap = cv2.VideoCapture(entry.get())

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    img = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img0 = Image.fromarray(cv2.resize(frame, (80, 60)))
            img0 = ImageTk.PhotoImage(image=img0)
            Label(tk, image=img0).pack()
            # convert the image to grayscale, blur it, and find edges
            # in the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply bilateral filter
            gray = cv2.bilateralFilter(gray, 9, 40, 40)

            # Canny edge detection
            edged = cv2.Canny(gray, 25, 50)
            # cv2.imshow('Canny', edged)

            # dilate borders
            dilate_kernel = np.ones((5, 5), np.uint8)
            edged = cv2.dilate(edged, dilate_kernel, iterations=2)
            # cv2.imshow('Canny + dilate', edged)

            rects, bounding_boxes = print_rectangles_with_findContours(edged.copy(), frame.copy())
            # cv2.imshow('Rectangles', rects)
            img1 = Image.fromarray(cv2.resize(rects, (80, 60)))
            img1 = ImageTk.PhotoImage(image=img1)
            Label(tk, image=img1).pack()

            img.append(rects)

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
tk.geometry("500x500")
tk.title("Video Analyzer")
label = Label(tk, text="Insert path to a video").pack()
entry = Entry()
entry.insert(0, "videos/VIRB0395.MP4")
entry.pack()
submit = Button(tk, text="Submit", command=analyze).pack()

tk.mainloop()
