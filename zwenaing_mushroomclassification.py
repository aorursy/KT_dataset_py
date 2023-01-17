# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')

dummies = pd.get_dummies(data)

X = dummies.drop(['class_e', 'class_p'], axis=1)

y = dummies['class_e']



train_X, test_X, train_y, test_y = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(train_X, train_y)

print("The training score is: {}\n".format(clf.score(train_X, train_y)))

print("The test score is: {}\n".format(clf.score(test_X, test_y)))
from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier()

clf.fit(train_X, train_y)

print("The training score is: {}\n".format(clf.score(train_X, train_y)))

print("The test score is: {}\n".format(clf.score(test_X, test_y)))
from sklearn.svm import SVC



clf = SVC(gamma=1)

clf.fit(train_X, train_y)

print("The training score is: {:.3f}\n".format(clf.score(train_X, train_y)))

print("The test score is: {:.3f}\n".format(clf.score(test_X, test_y)))
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(train_X, train_y)

print("The training score is: {:.3f}\n".format(clf.score(train_X, train_y)))

print("The test score is: {:.3f}\n".format(clf.score(test_X, test_y)))