import numpy as np


iou = [0.8147501200102073, 0.823127244567003, 0.8822576065854896, 0.9415498728059168, 0.9135807202871309,
       0.7864487585693898]

print(np.mean(iou))

tp = 32 + 36 + 9 + 32 + 33 + 19
print('tp: ', tp)

fp = 6 + 4 + 7 + 7 + 7
print('fp: ', fp)

fn = 8 + 26 + 5 + 3 + 2
print('fn: ', fn)

precision = [0.76, 0.8918918918918919, 0.8205128205128205, 1.0, 0.8372093023255814,
             0.8205128205128205]
print('precision --> ', np.mean(precision))

recall = [0.8, 0.5806451612903226, 1.0, 0.8648648648648649, 0.9166666666666666, 0.9047619047619048]
print('recall --> ', np.mean(recall))

f_measure = [0.8260869565217391, 0.9041095890410958, 0.8421052631578947, 1.0, 0.6857142857142857,
             0.810126582278481]
print('f-measure --> ', np.mean(f_measure))

b = 27 + 40 + 44 + 9 + 69 + 47
print('tot bb: ', b)
