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
from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
# load data

dataset = pd.read_csv('/kaggle/input/xgboost/Cell_000112.csv')



dataset.head()
dataset.shape
dataset = np.array(dataset)

dataset.shape
X=dataset[:,1:2]

Y=dataset[:,-1]
seed = 7

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train = X_train.astype(int)
y_train = y_train.astype(int)

X_test = X_test.astype(int)

y_test = y_test.astype(int)
# fit model no training data

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = np.array([round(value) for value in y_pred])
predictions
# evaluate predictions

y_test
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# load data

data = pd.read_csv('/kaggle/input/xgboost-test/test_Cell_000112.csv')

data = np.array(data)

X_data=data[:,1:2]

Y_data=data[:,-1]

X_data = X_data.astype(float)

Y_data = Y_data.astype(int)
y_data_pred = model.predict(X_data)
prediction_data = np.array([round(value) for value in y_data_pred])
accuracy = accuracy_score(Y_data, prediction_data)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
