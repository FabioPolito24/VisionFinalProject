import numpy as np
import cv2
import math


def print_rectangles_with_findContours(edged, frame):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = np.ones([len(contours), ])
    bounding_boxes = []
    # seleziono solo i contorni validi
    for i, contour in enumerate(contours):
        try:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            if w0 * h0 < 50000:
                # cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                rects[i] = 0
        except:
            rects[i] = 0
    # stampo i rettangoli che non sono contenuti dentro altri rettangoli
    for i, contour in enumerate(contours):
        if rects[i]:
            (x0, y0, w0, h0) = cv2.boundingRect(contour)
            for j, c in enumerate(contours):
                if rects[j]:
                    # se contour e c sono lo stesso contorno, vado avanti
                    if np.all(c == contour):
                        continue
                    else:
                        # controllo se contour è contenuto in c
                        (x1, y1, w1, h1) = cv2.boundingRect(c)
                        if x1 > x0 and y1 > y0 and x1 + w1 < x0 + w0 and y1 + h1 < y0 + h0:
                            # contour è contenuto in c quindi lo tolgo dalla lista
                            rects[i] = 0
                            break
            if rects[i] == 1:
                cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                bounding_boxes.append([x0, y0, w0, h0])
    return frame, bounding_boxes


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
