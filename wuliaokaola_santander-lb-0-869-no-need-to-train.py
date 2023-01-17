# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import scale

from sklearn.metrics import roc_auc_score
m1 = [81, 139, 12, 146, 76, 174, 21, 80, 166, 165, 13, 148, 198, 34, 115, 109, 44, 169, 149, 92, 108, 154, 33, 9, 192, 122, 121, 86, 123, 107, 127, 36, 172, 75, 177, 197, 87, 56, 93, 188, 131, 186, 141, 43, 104, 150, 31, 132, 23, 114, 58, 28, 116, 85, 194, 83]

m2 = [6, 110, 53, 26, 22, 99, 190, 2, 133, 0, 179, 1, 40, 184, 170, 78, 191, 94, 67, 18, 173, 118, 164, 89, 91, 147, 95, 35, 155, 106, 71, 157, 48, 162, 180, 163, 5, 145, 119, 32, 130, 49, 167, 90, 24, 195, 135, 151, 125, 128, 111, 52, 137, 70, 105, 51, 112, 199, 66, 82, 196, 175, 11, 74, 144, 8]

s = [26, 81, 139, 110, 12, 2, 22, 80, 53, 146, 179, 198, 99, 44, 0, 174, 76, 6, 166, 148, 133, 191, 40, 109, 190, 13, 123, 170, 165, 86, 108, 94, 21, 78, 1, 154, 184, 163, 91, 95, 75, 18, 93, 157, 89, 34, 119, 180, 115, 164, 92, 155, 9, 147, 56, 188, 122, 33, 130, 169, 5, 135, 51, 125, 141, 106, 151, 197, 162, 195, 172, 127, 121, 67, 111, 177, 173, 145, 132, 32, 43, 114, 131, 49, 36, 167, 88, 35, 107, 87, 175, 83, 149, 118, 196, 168, 150]
test_ds = pd.read_csv('../input/test.csv')

x = scale(test_ds.iloc[:, 1:])
sub = pd.DataFrame({"ID_code": test_ds.ID_code.values})

sub["target"] = np.std(x[:, s], axis=1) + np.mean(x[:, m2], axis=1) - np.mean(x[:, m1], axis=1)

sub.to_csv("submission.csv", index=False)