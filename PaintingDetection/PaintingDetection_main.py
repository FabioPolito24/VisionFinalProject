import cv2
import threading
import time
import os
import numpy as np
from cv2 import VideoWriter_fourcc
from tkinter import messagebox, Label, Entry, Button, Tk, Frame, LabelFrame
from PIL import Image, ImageTk
from PaintingDetection.detection_utils import *
from PaintingDetection.retrieval_utils import *
from PaintingDetection.rectification_utils import *
from PaintingDetection.general_utils import *
from PeopleLocalization.peopleLocalizator import *
from yolo.people_detector import *


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


class AnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.geometry("800x800")
        self.master.title("Video Analyzer")

        #--------- Instruction, text edit, and play button, all packed in one frame ---------
        self.play_frame = LabelFrame(self.master, text="Insert path to a video", padx=20, pady=10)
        self.play_frame.grid(row=0,column=0)
        #Path to video
        self.entry = Entry(self.play_frame)
        self.entry.insert(0, "../videos/vid001.MP4")
        self.entry.grid(row=0, column=0)
        #Play Button
        self.play_Button = Button(self.play_frame, text="Play", command=self.onThreadedClicked)
        self.play_Button.grid(row=0, column=1)

        #--------- Video container ---------
        self.rects_label_0 = Label(self.master, image="")
        self.rects_label_0.grid(row=1,column=0)
        #self.rects_label_1 = Label(self.master, image="")
        #self.rects_label_1.grid(row=2,column=0)

        # --------- Museum Map container ---------
        self.museum_map_label = Label(self.master, image="")
        self.museum_map_label.grid(row=2, column=0)

        # --------- Rectified paintings frame and container ---------
        # ToDo: print rectified images on GUI
        self.rect_paint_frame = LabelFrame(self.master, text="Rectified Paintings", padx=50, pady=50)
        self.rect_paint_frame.grid(row=0, column=1, rowspan=2)
        self.rectified_array = []
        self.array_max_lenght = 3

        # --------- People detector (YoloV3) ---------
        self.peopleDetector = PeopleDetector()

        # --------- Background Task ---------
        self.bg_task = BackgroundTask(self.analyze)

        # --------- DB paintings ---------
        self.db_paintings = load_db_paintings('../paintings_db/db_paintings.pickle')



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
        imgs = []
        counter = 0
        skip = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                skip -= 1
                if skip == 0:
                    skip = 1
                    scale_percent = 30
                    width = int(frame.shape[1] * scale_percent / 100)
                    height = int(frame.shape[0] * scale_percent / 100)
                    dsize = (width, height)

                    #Detect People inside actual frame
                    netOutput = self.peopleDetector.detectPeopleFromFrame(frame)

                    #Detect painting inside actual frame
                    frameWithBB, bounding_boxes0, rectified_images0 = first_step(method_1(frame.copy()), frame.copy(), self.db_paintings)
                    #img_1, bounding_boxes1, rectified_images1 = first_step(method_2(frame.copy()), frame.copy())

                    if len(netOutput.size()) > 0:
                        frameWithBB = self.peopleDetector.writLabels(frameWithBB, netOutput)

                    #Append actual frame to build a video at the end of execution
                    imgs.append(frameWithBB)

                    # Print actual frame on the gui
                    self.print_on_GUI(frameWithBB, self.rects_label_0, dsize)
                    # self.print_on_GUI(img_1, self.rects_label_1, dsize)

                    #Print Map with actual room
                    self.print_on_GUI(print_on_map(get_room("Sant'Antonio da Padova")), self.museum_map_label, dsize)

                    # uncomment the following lines when it's possibile to display rectified images un GUI
                    # for j, image in enumerate(rectified_images):
                    #     if j > self.array_max_lenght:
                    #         break
                    #     self.print_on_GUI(image, self.rectified_array[j], dsize)

                    for i, box in enumerate(bounding_boxes0):
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
            height, width, layers = imgs[1].shape
            fourcc = VideoWriter_fourcc(*'MP42')
            video = cv2.VideoWriter(folder_name + "/video.avi", fourcc, 24, (width, height))
            for j in range(len(imgs)):
                video.write(imgs[j])
            video.release()
        except:
            print('Video build failed')

        cap.release()
        cv2.destroyAllWindows()
        file.close()
        self.rects_label_0.configure(image="")
        self.rects_label_0.image = ""
        self.museum_map_label.configure(image="")
        self.museum_map_label.image = ""
        #self.rects_label_1.configure(image="")
        #self.rects_label_1.image = ""
        messagebox.showinfo("Info", "Video and csv file saved at location ./" + folder_name)



    def print_on_GUI(self, frame, label, dsize):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(frame, dsize))
        img = ImageTk.PhotoImage(image=img)
        label.forget()
        label.configure(image=img)
        label.image = img



if __name__ == "__main__":
    root = Tk()
    AnalyzerGUI(root)
    root.mainloop()
