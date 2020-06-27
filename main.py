import threading
import os
from cv2 import VideoWriter_fourcc
from tkinter import messagebox, Label, Entry, Button, Tk, LabelFrame
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
from PaintingDetection.detection_utils import *
from PaintingDetection.retrieval_utils import *
from PeopleLocalization.peopleLocalizator import *
from yolo.people_detector import *
from svm.ROI_classificator import *


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
            self.__bgTask_.stop()


class AnalyzerGUI:
    def __init__(self, master, width = 1200, height = 800):
        self.master = master
        self.window_width = width
        self.window_height = height
        self.master.geometry(str(width) + "x" + str(height))
        self.master.title("Video Analyzer")

        '''
        Gui format: 1200 x 800
        rectified.. and matched..  each 400x200
        video 400x400
        map 400x300
        ------------------------|-------------------------
        |   insert path         | rectified 1 | matched 1|
        |-----------------------|------------------------|
        |                       | rectified 2 | matched 2|
        |   video and labels    |------------------------|
        |-----------------------| rectified 3 | matched 3|
        |                       |------------------------|
        | map with localization | rectified 4 | matched 4|
        ------------------------|-------------------------
        '''

        # --------- Instruction, text edit, and play button, all packed in one frame ---------
        self.play_frame = LabelFrame(self.master, text="Insert path to a video", padx=20, pady=10)
        self.play_frame.grid(row=0,column=0)
        # Path to video
        self.entry = Entry(self.play_frame)
        self.entry.insert(0, "videos/vid001.MP4")
        self.entry.grid(row=0, column=0)
        # Play Button
        self.play_Button = Button(self.play_frame, text="Play", command=self.onThreadedClicked)
        self.play_Button.grid(row=0, column=1)

        # --------- Video container ---------
        self.out_video_dim = (int(self.window_width / 3), int(self.window_height / 2))
        self.video_label = Label(self.master, image="")
        frame = np.zeros((self.out_video_dim[1], self.out_video_dim[0], 3), dtype=np.uint8)
        img = Image.fromarray(frame, 'RGB')
        img = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=img)
        self.video_label.image = img
        self.video_label.grid(row=1,column=0)

        # --------- Museum Map container ---------
        self.museum_map_dim = (int(self.window_width / 3), int(self.window_height / 2))
        self.museum_map_label = Label(self.master, image="")
        frame = np.zeros((self.museum_map_dim[1], self.museum_map_dim[0], 3), dtype=np.uint8)
        img = Image.fromarray(frame, 'RGB')
        img = ImageTk.PhotoImage(image=img)
        self.museum_map_label.configure(image=img)
        self.museum_map_label.image = img
        self.museum_map_label.grid(row=2, column=0)

        # --------- Rectified paintings frame and container ---------
        # ToDo: print rectified images on GUI
        self.rect_paint_frame = LabelFrame(self.master, text="Rectified Paintings")
        self.rect_paint_frame.grid(row=0, column=1, rowspan=3)
        self.rectified_array = [Label(self.rect_paint_frame, image=""),Label(self.rect_paint_frame, image=""),Label(self.rect_paint_frame, image=""),Label(self.rect_paint_frame, image="")]
        self.max_num_rect_paint = 4
        self.rectified_dim = (int(self.window_width / 3), int(self.window_height / self.max_num_rect_paint))
        black_frame = np.zeros((self.rectified_dim[1], self.rectified_dim[0], 3), dtype=np.uint8)
        black_rectified_frame = Image.fromarray(black_frame, 'RGB')
        self.black_rectified_frame = ImageTk.PhotoImage(image=black_rectified_frame)
        for i in range(self.max_num_rect_paint):
            self.rectified_array[i].configure(image=self.black_rectified_frame)
            self.rectified_array[i].image = self.black_rectified_frame
            self.rectified_array[i].grid(row=i, column=0)

        # --------- Matched paintings frame and container ---------
        # ToDo: print matched images on GUI
        self.match_paint_frame = LabelFrame(self.master, text="Matched Paintings")
        self.match_paint_frame.grid(row=0, column=2, rowspan=3)
        self.matched_array = [Label(self.match_paint_frame, image=""),Label(self.match_paint_frame, image=""),Label(self.match_paint_frame, image=""),Label(self.match_paint_frame, image="")]
        self.max_num_matched_paint = 4
        self.matched_dim = (int(self.window_width / 3), int(self.window_height / self.max_num_matched_paint))
        black_frame = np.zeros((self.matched_dim[1], self.matched_dim[0], 3), dtype=np.uint8)
        black_matched_frame = Image.fromarray(black_frame, 'RGB')
        self.black_matched_frame = ImageTk.PhotoImage(image=black_matched_frame)
        for i in range(self.max_num_matched_paint):
            self.matched_array[i].configure(image=self.black_matched_frame)
            self.matched_array[i].image = self.black_matched_frame
            self.matched_array[i].grid(row=i, column=0)

        # --------- People detector (YoloV3) ---------
        self.peopleDetector = PeopleDetector(confidence=0.6)

        # --------- Background Task ---------
        self.bg_task = BackgroundTask(self.analyze)

        # --------- DB paintings initialization ---------
        PaintingsDB()

        # --------- SVM classifier initialization ---------
        clf_svm()

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
                os.mkdir("outputs")
            except:
                pass
            try:
                os.mkdir(folder_name)
            except:
                print("Directory for solution not created... Does it already exist?")
            file = open(folder_name + "/bounding_boxes.csv", 'w')
            file.write("frame,rect_id,x,y,w,h\n")
        except:
            print("Error creating file for solution")
            return

        # Calculate size for output video, preserving format
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if width > height:
                scale_percent = int(width / self.out_video_dim[1])
            else:
                scale_percent = int(height / self.out_video_dim[0])
            self.out_video_dim = (int(width / scale_percent), int(height / scale_percent))
            self.museum_map_dim = ( self.out_video_dim[0], self.out_video_dim[1])

        imgs = []
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:

                # Detect People inside actual frame
                netOutput = self.peopleDetector.detectPeopleFromFrame(frame)

                # Detect painting inside actual frame
                frameWithBB, bounding_boxes, rectified_images, paintings_matched = first_step(method_1(frame.copy()), frame.copy())

                # Put people localized on the frame
                if netOutput != None:
                    frameWithBB = self.peopleDetector.writLabels(frameWithBB, netOutput, bounding_boxes)

                # Print actual video frame on the gui
                self.print_on_GUI(frameWithBB, self.video_label, self.out_video_dim)

                # just some testing to create the db for svm
                # if frame_counter % 100 == 0:
                #     for box in bounding_boxes:
                #         label_hist(frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])

                # Print Map with actual room
                if len(paintings_matched) != 0:
                    id = os.path.basename(os.path.normpath(paintings_matched[0]['filename']))
                    self.print_on_GUI(print_on_map(get_room(id)), self.museum_map_label, self.museum_map_dim)
                else:
                    self.print_on_GUI(print_on_map(''), self.museum_map_label, self.museum_map_dim)

                # Print the matched paintings
                for j in range(self.max_num_matched_paint):
                    if j < len(paintings_matched):
                        self.print_on_GUI(paintings_matched[j]['im'], self.matched_array[j], self.matched_dim)
                    else:
                        self.print_on_GUI(self.black_matched_frame, self.matched_array[j], self.matched_dim)

                # display rectified images un GUI
                for j in range(self.max_num_rect_paint):
                    if j < len(rectified_images):
                        self.print_on_GUI(rectified_images[j], self.rectified_array[j], self.rectified_dim)
                    else:
                        self.print_on_GUI(self.black_rectified_frame, self.rectified_array[j], self.rectified_dim)

                for i, box in enumerate(bounding_boxes):
                    box_string = ""
                    for j in range(3):
                        box_string += str(box[j]) + ","
                    box_string += str(box[3])
                    file.write(str(frame_counter) + "," + str(i) + "," + box_string + "\n")

                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break

                # Append actual frame to create a video at the end
                imgs.append(self.get_screenshot())

                frame_counter += 1

            else:
                break

        # Try to build a video of the entire video analyzer
        try:
            height, width, layers = imgs[1].shape
            fourcc = VideoWriter_fourcc(*'MP42')
            video = cv2.VideoWriter(folder_name + "/video.avi", fourcc, 15, (width, height))
            for j in range(len(imgs)):
                video.write(imgs[j])
            video.release()
            messagebox.showinfo("Info", "Video and csv file saved at location ./" + folder_name)
        except:
            print('Video build failed')

        # Close interface
        cap.release()
        file.close()
        self.delete_GUI_imgs()
        cv2.destroyAllWindows()

    def print_on_GUI(self, frame, label, out_video_dim):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(frame, out_video_dim))
            img = ImageTk.PhotoImage(image=img)
            label.forget()
            label.configure(image=img)
            label.image = img
        except:
            frame = np.zeros((self.rectified_dim[1], self.rectified_dim[0], 3), dtype=np.uint8)
            img = Image.fromarray(frame, 'RGB')
            img = ImageTk.PhotoImage(image=img)
            label.forget()
            label.configure(image=img)
            label.image = img

    def delete_GUI_imgs(self):
        self.video_label.forget()
        self.museum_map_label.forget()
        for i in range(self.max_num_rect_paint):
            self.rectified_array[i].forget()
        for i in range(self.max_num_matched_paint):
            self.matched_array[i].forget()

    def get_screenshot(self):
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        xx = x + self.master.winfo_width()
        yy = y + self.master.winfo_height()
        return cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x, y, xx, yy)).convert('RGB')), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    root = Tk()
    AnalyzerGUI(root)
    root.mainloop()
