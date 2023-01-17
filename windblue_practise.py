# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt
import seaborn as sns
X_train = train.drop('label',axis=1)
y_train = train.loc[:,'label']
from sklearn.svm import LinearSVR
model = LinearSVR()
model.fit(X_train,y_train)
y_test = model.predict(test)
y_test.shape
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['Label'] = y_test
sample_submission.to_csv('./change.csv',index=False)
