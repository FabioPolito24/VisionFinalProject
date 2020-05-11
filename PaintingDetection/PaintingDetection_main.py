import cv2
import threading
import time
import os
import numpy as np
from cv2 import VideoWriter_fourcc
from tkinter import messagebox, Label, Entry, Button, Tk
from PIL import Image, ImageTk
from utils import *


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
                messagebox.showerror("Error", repr(e))
                # pass
            self.__bgTask_.stop()


def tkThreadingTest():
    class AnalyzerGUI:
        def __init__(self, master):
            self.master = master
            self.master.geometry("600x700+400+180")
            self.master.title("Video Analyzer")
            Label(self.master, text="Insert path to a video").pack()
            self.entry = Entry()
            self.entry.insert(0, "../videos/vid001.MP4")
            self.entry.pack()
            Button(root, text="Submit", command=self.onThreadedClicked).pack()
            self.rects_label_0 = Label(self.master, image="")
            self.rects_label_0.pack()
            self.rects_label_1 = Label(self.master, image="")
            self.rects_label_1.pack()

            self.bg_task = BackgroundTask(self.analyze)

        def onThreadedClicked(self):
            try: self.bg_task.start()
            except: pass

        def analyze(self, isRunningFunc=None):
            cap = cv2.VideoCapture(self.entry.get())

            # Check if camera opened successfully
            if not cap.isOpened():
                print("Error opening video stream or file")
                messagebox.showerror("Error", "Video not found")
                return
            try:
                name = self.entry.get()
                name = name.split("/")[-1]
                name = name.split(".")[0]
                folder_name = "outputs/" + name
                try:
                    os.mkdir(folder_name)
                except:
                    print("Directory for solution not created... Does it already exist?")
                file = open(folder_name + "/bounding_boxes.csv", 'w')
                file.write("frame,rect_id,x,y,w,h\n")
            except:
                print("Error creating file for solution")
                return
            img = []
            counter = 0
            b_hist, g_hist, r_hist = get_mean_hist()
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    scale_percent = 30
                    width = int(frame.shape[1] * scale_percent / 100)
                    height = int(frame.shape[0] * scale_percent / 100)
                    dsize = (width, height)

                    rects_0, bounding_boxes = print_rectangles_with_findContours(method_0(frame.copy()), frame.copy(),
                                                                                 b_hist, g_hist, r_hist)
                    rects_1, bounding_boxes = print_rectangles_with_findContours(method_1(frame.copy()), frame.copy(),
                                                                                 b_hist, g_hist, r_hist)
                    # cv2.imshow('Rectangles', rects)
                    img.append(rects_0)

                    self.print_on_GUI(rects_0, self.rects_label_0, dsize)
                    self.print_on_GUI(rects_1, self.rects_label_1, dsize)

                    for i, box in enumerate(bounding_boxes):
                        box_string = ""
                        for j in range(3):
                            box_string += str(box[j]) + ","
                        box_string += str(box[3])
                        file.write(str(counter) + "," + str(i) + "," + box_string + "\n")
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break
                    # sleep because otherwise frames are displayed too rapidly
                    # time.sleep(0.02)
                    counter += 1

                else:
                    break

            try:
                height, width, layers = img[1].shape
                fourcc = VideoWriter_fourcc(*'MP42')
                video = cv2.VideoWriter(folder_name + "/video.avi", fourcc, 24, (width, height))
                for j in range(len(img)):
                    video.write(img[j])
                video.release()
            except:
                print('Video build failed')

            cap.release()
            cv2.destroyAllWindows()
            file.close()
            self.rects_label_0.configure(image="")
            self.rects_label_0.image = ""
            self.rects_label_1.configure(image="")
            self.rects_label_1.image = ""
            messagebox.showinfo("Info", "Video and csv file saved at location ./" + folder_name)

        def print_on_GUI(self, rects, label, dsize):
            rects = cv2.cvtColor(rects, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(rects, dsize))
            img = ImageTk.PhotoImage(image=img)
            label.configure(image=img)
            label.image = img

    root = Tk()
    AnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    tkThreadingTest()
