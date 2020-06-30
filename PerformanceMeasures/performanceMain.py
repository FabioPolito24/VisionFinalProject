import pandas as pd
import numpy as np
import glob


def IoU(bb1, bb2, frameW, frameH):
    x1 = bb1['x'] * frameW
    y1 = bb1['y'] * frameH
    x2 = bb2['x'] * frameW
    y2 = bb2['y'] * frameH
    w1 = bb1['width'] * frameW
    h1 = bb1['height'] * frameH
    w2 = bb2['width'] * frameW
    h2 = bb2['height'] * frameH
    totalArea = w1 * h1 + w2 * h2
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the total area covered by the two boxes
    boxArea = totalArea - interArea
    # compute the intersection over union
    iou = interArea / float(boxArea)
    return iou


frameW = 1280
frameH = 720
files = glob.glob("vid01/*.txt")
iou_list = []
tp = 0
fp = 0
fn = 0

for file in files:
    df = pd.read_csv(file, sep=' ', header=None)
    df.columns = ["id", "x", "y", "width", "height"]
    ground_truth_start = df.index[df['id'] == 0].tolist()[1]

    for i in range(df['id'].max() + 1):
        ids = df.loc[df['id'] == i]
        if len(ids) == 2:
            bb1 = ids.iloc[0].drop('id')
            bb2 = ids.iloc[1].drop('id')
            iou_list.append(IoU(bb1, bb2, frameW, frameH))
            tp += 1
        else:
            if i > ground_truth_start:
                fn += 1
            else:
                fp += 1


precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_measure = 2 * precision * recall / (precision + recall)
print('Average IoU: ', np.mean(iou_list))
print('FP: ', fp)
print('FN: ', fn)
print('TP: ', tp)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-Measure: ', f_measure)
