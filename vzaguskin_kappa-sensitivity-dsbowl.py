import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import cohen_kappa_score, accuracy_score

import random

random.seed(42)

np.random.seed(42)

gt = np.random.choice([0,1,2,3], 1000, p=(0.24, 0.14, 0.12, 0.50))

pred = gt.copy()

noise = np.random.choice([-2, -1, 0, 1, 2], 1000, p=(0.1, 0.2, 0.4, 0.2, 0.1))

pred_noisy = pred + noise

pred_noisy[pred_noisy > 3]  -=3

pred_noisy[pred_noisy < 0] += 3
cohen_kappa_score(gt, pred_noisy, weights='quadratic'), accuracy_score(gt, pred_noisy)
for i in range(10):

    errs = np.where(pred_noisy != gt)[0]

    pred_noisy[errs[:10]] = gt[errs[:10]]

    print("kappa: {}, acc: {}".format(cohen_kappa_score(gt, pred_noisy, weights='quadratic'), accuracy_score(gt, pred_noisy)))

    