import pandas
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from svm.ROI_classificator import *

# Add the data of the roi to the data file used to train the SVM
def label_hist(roi):
    h = create_hist(roi)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imshow('calcHist Demo', histImage)
    # cv2.waitKey()
    label = input("Painting? (0:'no', 1:'yes') -->  ")
    row = pd.DataFrame(h)
    row['label'] = label
    row.to_csv('../svm/data.csv', sep=',', mode='a', header=False)
