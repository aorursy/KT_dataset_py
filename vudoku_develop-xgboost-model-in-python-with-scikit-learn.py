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
import scipy

print('scipy: {}'.format(scipy.__version__))
import xgboost as xgb

print('XGBoost: {}'.format(xgb.__version__))

xgb.__version__
from numpy import loadtxt

import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# load data

#pima_data = pd.read_csv("../input/pima-indians-diabetes.data.csv")

pima_data = loadtxt("../input/pima-indians-diabetes.data.csv", delimiter = ",")

#print(pima_data.head(5))
#split data into X and Y

X = pima_data[:,0:8]

Y = pima_data[:,8]
# split data into train and test sets

seed = 7

test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
# fit model no training data

model = XGBClassifier()

model.fit(X_train, Y_train)
print(model)
# make predictions for test data

Y_pred = model.predict(X_test)

predictions = [round(value) for value in Y_pred]
# evaluate predictions

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%"%(accuracy * 100.0))