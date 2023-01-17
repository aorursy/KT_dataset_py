# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
hr = pd.read_csv("../input/HR_comma_sep.csv")
hr.head()
hr.groupby(["number_project", "left"]).count()
hr["sales"].unique()
hr["salary"].unique()
hr_X = hr.loc[:, hr.columns != "left"]

hr_X_one_hot_encoding = pd.get_dummies(hr_X)

hr_y = hr["left"]
hr_X_one_hot_encoding.head()
from sklearn.model_selection import train_test_split

hr_X_one_hot_train, hr_X_one_hot_test, hr_y_train, hr_y_test = train_test_split(hr_X_one_hot_encoding, hr_y)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(hr_X_one_hot_train, hr_y_train)

y_pred = clf.predict(hr_X_one_hot_test)
clf.feature_importances_
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

print(classification_report(hr_y_test, y_pred))

print(confusion_matrix(hr_y_test, y_pred))