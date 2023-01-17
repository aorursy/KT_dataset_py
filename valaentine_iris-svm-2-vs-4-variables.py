# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import svm, datasets

from sklearn.model_selection import train_test_split



iris = datasets.load_iris()

# print(iris.data)



X = iris.data[:, :2]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



models = (svm.SVC(kernel='linear', C=1.0),

          svm.LinearSVC(C=1.0, max_iter=10000),

          svm.SVC(kernel='rbf', gamma=0.7, C=1.0),

          svm.SVC(kernel='poly', degree=3, gamma='auto', C=1.0))

models = (clf.fit(X_train, y_train) for clf in models)



for clf in models:

    print(clf.score(X_test,y_test))

X = iris.data[:, :4]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

models = (svm.SVC(kernel='linear', C=1.0),

          svm.LinearSVC(C=1.0, max_iter=10000),

          svm.SVC(kernel='rbf', gamma=0.7, C=1.0),

          svm.SVC(kernel='poly', degree=3, gamma='auto', C=1.0))

models = (clf.fit(X_train, y_train) for clf in models)





for clf in models:

    print(clf.score(X_test,y_test))


