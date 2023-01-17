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
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from datetime import datetime

import time

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
scaler = MinMaxScaler()

train_labels = train['label']

scaled_features = scaler.fit_transform(train.drop(['label'], axis=1))

X_train, X_dev, y_train, y_dev = train_test_split(scaled_features, train_labels, test_size=0.20)
print(X_train.shape)

print(y_train.shape)

xgb = XGBClassifier(n_estimators=100, nthread=4)



training_start = time.perf_counter()

xgb.fit(X_train, y_train)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds = xgb.predict(X_dev)

prediction_end = time.perf_counter()

xgb_train_time = training_end-training_start

xgb_prediction_time = prediction_end-prediction_start

print("Time consumed for training set: %4.3f" % (xgb_train_time))

print("Time consumed for dev set: %6.5f seconds" % (xgb_prediction_time))

target_names = ['0','1','2','3','4','5','6','7','8','9']

report = classification_report(y_dev, preds, target_names=target_names, digits=3)

print(report)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(test.shape)

#test_ndarray = test.values

#print(type(test_ndarray))

scaled_test = scaler.fit_transform(test)

test_start = time.perf_counter()

y_hat = xgb.predict(scaled_test)

test_end = time.perf_counter()

xgb_test_time = test_end-test_start

print("Time consumed for testing set: %4.3f" % (xgb_test_time))
sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_sub.head()
y_hat = pd.Series(y_hat, name='label')

ids = np.arange(len(y_hat)) + 1

ids = pd.Series(ids, name='ImageId')

submission = pd.concat([ids,y_hat], axis=1)

submission.to_csv('/kaggle/working/submission.csv', index=False)

print(submission.head())

print(submission.describe())

print("Done")