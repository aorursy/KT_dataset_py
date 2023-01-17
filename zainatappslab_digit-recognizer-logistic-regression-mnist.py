# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

print('starting to run')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


datasetTrain = pd.read_csv('../input/train.csv').as_matrix()
X = datasetTrain[:, 1:]
y = datasetTrain[:, 0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=8)

datasetTest = pd.read_csv('../input/test.csv').as_matrix()

clf = LogisticRegression(verbose=True, max_iter=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(datasetTest)

df = pd.DataFrame(data={'ImageId': list(range(1,28001)), 'Label':y_pred});
df.to_csv('mnistsubmissionfinal.csv', index = False)
print('score = {} '.format(clf.score(X_test, y_test)))

# Any results you write to the current directory are saved as output.
!ls