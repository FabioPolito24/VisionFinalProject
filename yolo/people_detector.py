from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

class PeopleDetector:
    def __init__(self, ):
        self.confidence = 0.5
        self.nms_thesh = 0.4
        self.CUDA = torch.cuda.is_available()
        self.weightsfile = "yolov3.weights"
        self.cfgfile = "cfg/yolov3.cfg"
        self.reso = 416
        #Set up NN:
        self.num_classes = 80
        self.classes = load_classes("data/coco.names")
        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        print("Network successfully loaded")
        self.model.net_info["height"] = self.reso
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        # If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()
        # Set model in evaluation mode
        self.model.eval()

    def prep_image(self, img):
        """
        Prepare image for inputting to the neural network.
        """
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (self.inp_dim, self.inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def write(self, x, img):
        """
        Put label on top of image
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        colors = pkl.load(open("pallete", "rb"))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img

    def detectPeopleFromFrame(self, frame):
        #Prepare imgs compatible with pytorch
        img, orig_im, dim = self.prep_image(frame)

        #Load img on GPU if available
        if self.CUDA:
            img = img.cuda()

        #Inference time
        with torch.no_grad():
            output = self.model(Variable(img), self.CUDA)

        #Collect 3 stage prediction into single one
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        #If No detection...
        if type(output) == int:
            cv2.imshow("frame int", orig_im)
            cv2.waitKey()

        #If we have detection mantain only people --> people id == 0
        output = output[output[:,-1] < 1]

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        list(map(lambda x: self.write(x, orig_im), output))

        cv2.imshow("frame", orig_im)
        cv2.waitKey()


def main():
    """
        Example usage of this class:

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        det.detectPeopleFromFrame(frame)
    """
    det = PeopleDetector()
    example_img = cv2.imread('imgs/messi.jpg')
    det.detectPeopleFromFrame(example_img)

if __name__ == "__main__":
    main()