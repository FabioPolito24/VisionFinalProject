# VisionFinalProject
### Painting detection: predict a ROI for each painting
- Given an input video, your code should output a list of bounding boxes (x, y, w, h), being (x,y) the upper-left corner, each containing one painting.
- Create an interface to visualize given an image the ROI of a painting.
- Select painting and discard other artifacts.
- (optional: segment precisely paintings with frames and also statues)
### Painting rectification: correct the perspective distortion of each painting
- Given an input video and your detections (from the previous point), your code should output a new image for each painting, containing the rectified version of the painting.
- be careful on not squared painting
### Painting retrieval: match each detected painting to the paintings DB
- Given one rectified painting (from the previous point), your code should return a ranked list of all the images in the painting DB, sorted by descending similarity with the detected painting. Ideally, the first retrieved item should be the picture of the detected painting.
### People detection: predict a ROI around each person
- Given an input video, your code should output a list of bounding boxes (x, y, w, h), being (x,y) the upper-left corner, each containing one person. 
- People localization: assign each person to one room on the map
- Given an input video and people bounding boxes (from the previous point), your code should assign each person to one of the rooms of the Gallery. To do that, you can exploit the painting retrieval procedure (third point), and the mapping between paintings and rooms (in “data.csv”). Also, a map of the Gallery is available (“map.png”) for a better visualization.

## Optional tasks
### Determine whether each person is facing a painting
- Given an input video, people and paintings' detections, determine whether each person is facing a painting or not.
### Replace paintings areas in the 3D model with the provided pictures
- Given a view taken from the 3D model, detect each painting and replace it with its corresponding picture in the paintings DB, appropriately deformed to match the 3D view.
### Determine the distance of a person to the closest door
- Find the door, find the walls and the floor, try to compensate and predict distance

## PaintingDetection Pipeline
1. Converting From rgb to gray each frame
2. Apply Bilateral filter
3. Edge finding with Canny
4. Dilate edges to detect better contours
5. Find contours with Opencv standard function
6. Filter Contour to obtain only the external ones

## PaintingRectification Pipeline
1. TODO...

## PaintingRetrival Pipeline
1. TODO...

## People Detection Pipeline
1. Detection made with YoloV3
How to add it to the project:
- Include Yolo/people_detector.py in yout project
- Initialize (once) the network: detector = PeopleDetector()
- To detect people call method passing a simple frame from videocapture frame: det.detectPeopleFromFrame(frame)

## Progect structure:
    .
    ├── PaintingDetection
    │   ├── outputs            		# Video edited with bounding box
    │   │   └── ...	       		# videos...
    │   ├── PaintingDetection_main.py   # Main of Painting detection
    │   ├── detection_utils.py          # detection functions
    │   ├── rectification_utils.py      # rectification functions
    │   ├── retrival_utils.py         	# retrival functions
    │   └── ...
    ├── painting_db 	       		# Database of all Paintings
    ├── videos 	       			# Example videos ready to use
    └── yolo		       		# YoloV3 NN used for people detection
	├── ...            		# Other utilities for yolo
	├── util.py			# Some methods usefull
	├── people_detector.py		# Class to make inference in video frames
	└── yolov3.weights		# Download Weights: https://pjreddie.com/media/files/yolov3.weights
