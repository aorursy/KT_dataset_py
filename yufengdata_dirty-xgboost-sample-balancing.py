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
train = pd.read_csv("../input/train.csv", nrows = 10000000)
train.head()
test = pd.read_csv("../input/test.csv")
test.head()
import matplotlib.pyplot as plt
%matplotlib inline
count_classes = pd.value_counts(train['is_attributed'],sort=True).sort_index()
count_classes.plot(kind='bar')
plt.title("class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
y_train = train['is_attributed']

x_train = train[['ip', 'app', 'device', 'os', 'channel']]
x_test  = test[['ip', 'app', 'device', 'os', 'channel']]

del train
del test

from imblearn.over_sampling import SMOTE
sm = SMOTE()
x_train, y_train = sm.fit_sample(x_train, y_train)
print("Finish balance the data, Finish normalize the data")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

X_test = pd.DataFrame(x_test, columns = ['f0', 'f1', 'f2', 'f3', 'f4'])
y_pred = model.predict_proba(X_test)
sub = pd.read_csv("../input/sample_submission.csv")

sub['is_attributed'] = y_pred[:,0]
sub.head()
sub.to_csv('sub_rf.csv', index=False)

