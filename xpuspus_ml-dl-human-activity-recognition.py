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
from math import sqrt

from numpy import concatenate

from matplotlib import pyplot as plt

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
import keras

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix as cf

from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

from keras.optimizers import Adam
from keras.utils import vis_utils

from keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier as RF

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix as cf

# from keras.utils.np_utils import to_categorical

from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score as cv_score

import time

from sklearn.metrics import roc_curve as roc

from sklearn.metrics import auc as auc
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head(2)
test_data.head(2)
X_train = train_data.drop('Activity', axis = 1)
y_train = pd.get_dummies(train_data.Activity)
X_test = test_data.drop('Activity', axis = 1)
y_test = pd.get_dummies(test_data.Activity)
rf = RF(20)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cf(np.argmax(y_test.as_matrix(), axis = 1), np.argmax(y_pred, axis = 1))
rf.score(X_test, y_test)
plt.matshow(cf(np.argmax(y_test.as_matrix(), axis = 1), np.argmax(y_pred, axis = 1)), cmap = 'Reds')