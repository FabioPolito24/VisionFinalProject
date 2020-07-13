# VisionFinalProject
## How to use this code:
1. Install requirements
2. Download Weights for yoloV3 : https://pjreddie.com/media/files/yolov3.weights
3. Put them in yolo/weights as yolov3.weights
4. Run save_key_points.py to create local db of keypoints
5. Run main.py and Enjoi the project, at the end of the process a file will be created in output/ folder
6. Run ReplacingPaintings3dModel_main.py to replace paintings in 3d model

## Description of the functionality:
### Painting detection: predict a ROI for each painting
- Given an input video, your code should output a list of bounding boxes (x, y, w, h), being (x,y) the upper-left corner, each containing one painting.
- Create an interface to visualize given an image the ROI of a painting.
- Select painting and discard other artifacts.
- (optional: segment precisely paintings with frames)
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
### Replace paintings areas in the 3D model with the provided pictures
- Given a view taken from the 3D model, detect each painting and replace it with its corresponding picture in the paintings DB, appropriately deformed to match the 3D view.

## PaintingDetection Pipeline
1. Converting From rgb to gray each frame
2. Apply Bilateral filter
3. Otsu thresholding
4. Dilate edges to detect better contours
5. Find contours with Opencv standard function
6. Filter Contour to obtain only the external ones

## PaintingRectification Pipeline
### Method 1
1. ApproxPolydp
2. If we have exactly 4 vertexies continue
3. Calculate Homography matrix for a standard rectangle
4. WarpPerspective to rectifie the paintings
### Method 2
1. If we have a good painting retrived from db
2. Use keypoints calculated to compute Homography matrix
3. WarpPerspective to rectifie the paintings

## PaintingRetrival Pipeline
0. With ORB we previously calculated and stored keypoints using the pickle module for all db paintings
1. Keypoint detection in actual Bounding box with ORB
2. Ratio test ratio test proposed by D. Lowe in the SIFT paper is performed on all db painting
3. If ratio i greater than a threshold is a good match and continue
4. Save best 5 match in a csv file
5. Display only the best one on the gui


## People Detection Pipeline
1. Detection made with YoloV3 trained on COCO dataset.
2. If the network dont'find any persons will return None
3. Before print all bb a check is done with prevuosly painting founded,if a person is inside a paint discard it
4. Otherwise print bounding bosex inn red with function: writLabels()
### People Localization
1. If we have a good painting matched in db continue
2. Find painting position on museum map
3. Assuming people in actual frame is in the same room of the painting
4. Print a red dot in the selected room

## Progect structure:
    .
    ├── main.py                            # main program to detect paintings
    ├── ReplacingPaintings3dModel_main.py  # main program to relpace paintings in 3d model
	├── save_key_points.py                 # script to generate keypoits db
    ├── outputs                            # Video edited with bounding box
    │   └── ...                            # videos created by gui
    ├── PaintingDetection
    │   ├── general_utils.py               # general functions
    │   ├── detection_utils.py             # detection functions
    │   ├── rectification_utils.py         # rectification functions
    │   ├── retrival_utils.py              # retrival functions
    │   ├── histogram.py                   # histogram
    │   └── ...
    ├── painting_db                        # Database of all Paintings
    ├── videos                             # Example videos ready to use
    ├── PeopleLocalization                 # People localization on image map
    │   ├── images
    │   │   └── map.png                    # map of the museum
    │   └── peopleLocalizator.py           # Script that print red dot on actual room
    ├── SVM                                # SVM to determine if a bb contains a painting or not
    │   ├── dbCreator.py
    │   └── ROI_classificator.py
    ├── PerformanceMesures                 # Frame with hand made label and automatically generated once
    └── yolo                               # YoloV3 NN used for people detection
        ├── ...                            # Other utilities for yolo
        ├── weights                        # Folder containing weights
        │   └── yolov3.weights             # Download Weights: https://pjreddie.com/media/files/yolov3.weights
        ├── util.py                        # Some methods usefull
        └── people_detector.py             # Class to make inference in video frames
