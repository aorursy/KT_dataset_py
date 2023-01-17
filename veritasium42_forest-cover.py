# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from datetime import datetime

import time

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

#train.info()

#train['Cover_Type'].value_counts()

#train.columns.values

#train.values
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

#test.info()

#test.columns.values

type(test)
labels = train['Cover_Type']



scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(train.drop(['Id', 'Cover_Type'], axis=1))

X_train, X_dev, y_train, y_dev = train_test_split(scaled_features, labels, test_size=0.20)

#X_train, X_dev, y_train, y_dev = train_test_split(train.drop(['Cover_Type', 'Id'], axis=1), labels, test_size=0.30)

xgb = XGBClassifier(n_estimators=500, nthread=4)



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

target_names = ['Type-1','Type-2','Type-3','Type-4','Type-5','Type-6','Type-7']

report = classification_report(y_dev, preds, target_names=target_names, digits=3)

print(report)
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

#test.info()

#test.columns.values

ids = test['Id']

test = test.drop(['Id'], axis=1)

#test_ndarray = test.values

#print(type(test_ndarray))

scaled_test = scaler.fit_transform(test)

test_start = time.perf_counter()

y_hat = xgb.predict(scaled_test)

test_end = time.perf_counter()

xgb_test_time = test_end-test_start

print("Time consumed for testing set: %4.3f" % (xgb_test_time))

y_hat = pd.Series(y_hat, name='Cover_Type')

ids = pd.Series(ids, name='Id')

submission = pd.concat([ids,y_hat], axis=1)

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Done")