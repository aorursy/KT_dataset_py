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
dataset=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print("Keys of dataset:\n{}".format(dataset.keys()))

print("{}".format(dataset))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['label'],random_state=0)

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

X_train.shape
y_pred=knn.predict(X_test)

print("Test set predictions:\n{}".format(y_pred))
y_pred=knn.predict(X_test)

print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))