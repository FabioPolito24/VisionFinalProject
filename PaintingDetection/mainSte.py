import cv2
import threading
import time
import numpy as np
from cv2 import VideoWriter_fourcc
from tkinter import messagebox, Label, Entry, Button, Tk
from PIL import Image, ImageTk
from utils import print_rectangles_with_findContours


class BackgroundTask:
    def __init__(self, taskFuncPointer):
        self.__taskFuncPointer_ = taskFuncPointer
        self.__workerThread_ = None
        self.__isRunning_ = False

    def taskFuncPointer(self): return self.__taskFuncPointer_

    def isRunning(self):
        return self.__isRunning_ and self.__workerThread_.isAlive()

    def start(self):
        if not self.__isRunning_:
            self.__isRunning_ = True
            self.__workerThread_ = self.WorkerThread(self)
            self.__workerThread_.start()

    def stop(self): self.__isRunning_ = False

    class WorkerThread(threading.Thread):
        def __init__(self, bgTask):
            threading.Thread.__init__(self)
            self.__bgTask_ = bgTask

        def run(self):
            try:
                self.__bgTask_.taskFuncPointer()(self.__bgTask_.isRunning)
            except Exception as e:
                # messagebox.showerror("Error", repr(e))
                pass
            self.__bgTask_.stop()


def tkThreadingTest():
    class AnalyzerGUI:
        def __init__(self, master):
            self.master = master
            self.master.geometry("600x400+180+180")
            self.master.title("Video Analyzer")
            Label(self.master, text="Insert path to a video").pack()
            self.entry = Entry()
            self.entry.insert(0, "videos/VIRB0395.MP4")
            self.entry.pack()
            Button(root, text="Submit", command=self.onThreadedClicked).pack()
            self.rects_label = Label(self.master, image="")
            self.rects_label.pack()

            self.bg_task = BackgroundTask(self.analyze)

        def onThreadedClicked(self):
            try: self.bg_task.start()
            except: pass

        def analyze(self, isRunningFunc=None):
            cap = cv2.VideoCapture(self.entry.get())

            # Check if camera opened successfully
            if not cap.isOpened():
                print("Error opening video stream or file")

            img = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    scale_percent = 40
                    width = int(frame.shape[1] * scale_percent / 100)
                    height = int(frame.shape[0] * scale_percent / 100)
                    dsize = (width, height)

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
                    img.append(rects)
                    rects = cv2.cvtColor(rects, cv2.COLOR_BGR2RGB)
                    img1 = Image.fromarray(cv2.resize(rects, dsize))
                    img1 = ImageTk.PhotoImage(image=img1)
                    # self.rects_label["image"] = img1
                    self.rects_label.configure(image=img1)
                    self.rects_label.image = img1

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    # sleep because otherwise frames are displayed too rapidly
                    # time.sleep(0.02)

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
            self.rects_label.configure(image="")
            self.rects_label.image = ""

    root = Tk()
    AnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    tkThreadingTest()
