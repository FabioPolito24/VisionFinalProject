from __future__ import division
from yolo.util import *
from yolo.darknet import Darknet

'''
    Example usage of this class:

    det = PeopleDetector()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    output = det.detectPeopleFromFrame(frame)
    if netOutput != None:
        frameWithBB = det.writLabels(frame, netOutput)
'''

class PeopleDetector:
    def __init__(self, confidence = 0.5, nms_thresh = 0.4, resolution = 416, weights_path = '../yolo/weights/yolov3.weights', cfg_path = '../yolo/cfg/yolov3.cfg', num_classes = 80, names_path = '../yolo/data/coco.names'):
        self.confidence = confidence
        self.nms_thesh = nms_thresh
        self.weightsfile = weights_path
        self.cfgfile = cfg_path
        self.CUDA = torch.cuda.is_available()
        self.num_classes = num_classes
        self.classes = load_classes(names_path)
        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = resolution
        self.inp_dim = int(self.model.net_info["height"])
        #Check if resolution is multiple of 32
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        # If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()
        # Set model in evaluation mode
        self.model.eval()

    def prep_image(self, img):
        """
        Prepare image (resize) for inputting to the neural network.
        """
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (self.inp_dim, self.inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def writeSingleLabel(self, x, img, color = (0, 0, 255)):
        """
        Put label on top of image
        Default label color: red
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img

    def writLabels(self, origin_im, netOutput):
        """
        Put all the labels on top of image
        """
        list(map(lambda x: self.writeSingleLabel(x, origin_im), netOutput))
        return origin_im

    def detectPeopleFromFrame(self, frame):
        """
        Detect people inside a frame and return bounding boxes
        """
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
            return None

        #If we have detection maintain only people --> people id == 0
        output = output[output[:,-1] < 1]

        #Resize Label according to input frame dimension
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        return output