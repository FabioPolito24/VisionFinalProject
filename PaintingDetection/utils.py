import numpy as np
import cv2
import math


def print_rectangles_with_findContours(edged, frame):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        try:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > 5000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except:
            pass

    return frame


def houghLines(edged):
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 50, None, 0, 0)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(edged, pt1, pt2, (255, 255, 255), 3)
    return edged


def isolate_painting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_frame_color = np.array([15, 19, 24])
    upper_frame_color = np.array([110, 143, 171])
    mask = cv2.inRange(hsv, lower_frame_color, upper_frame_color)
    return mask
