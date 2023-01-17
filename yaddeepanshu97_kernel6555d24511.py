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


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm



import keras

from pykalman import KalmanFilter

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, recall_score, precision_score

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Activation



np.random.seed(1234)  

PYTHONHASHSEED = 0

%matplotlib inline


# read training data 

train_df = pd.read_csv('../input/train_FD001.txt', sep=" ", header=None)

train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)

train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',

                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',

                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
# read test data

test_df = pd.read_csv('../input/test_FD001.txt', sep=" ", header=None)

test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)

test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',

                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',

                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
# read ground truth data

truth_df = pd.read_csv('../input/RUL_FD001.txt', sep=" ", header=None)

truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
train_df = train_df.sort_values(['id','cycle'])

train_df.head()


# Data Labeling - generate column RUL

rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()

rul.columns = ['id', 'max']

train_df = train_df.merge(rul, on=['id'], how='left')

train_df['RUL'] = train_df['max'] - train_df['cycle']

train_df.drop('max', axis=1, inplace=True)

train_df.head()
# generate label columns for training data

w1 = 30

w0 = 24

train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )

train_df['label2'] = train_df['label1']

train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

train_df.head()
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()

rul.columns = ['id', 'max']

truth_df.columns = ['more']

truth_df['id'] = truth_df.index + 1

truth_df['max'] = rul['max'] + truth_df['more']

truth_df.drop('more', axis=1, inplace=True)
# generate RUL for test data

test_df = test_df.merge(truth_df, on=['id'], how='left')

test_df['RUL'] = test_df['max'] - test_df['cycle']

test_df.drop('max', axis=1, inplace=True)

test_df.head()
# generate label columns w0 and w1 for test data

test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )

test_df['label2'] = test_df['label1']

test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

test_df.head()
# pick the feature columns 

sensor_cols = ['s' + str(i) for i in range(1,22)]

cols = ['setting1', 'setting2', 'setting3']

cols.extend(sensor_cols)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import lightgbm as lgb



X_train, X_val, Y_train, Y_val = train_test_split(train_df[cols], train_df['label1'], test_size=0.05, 

                                                  shuffle=False, random_state=42)



print ("Train_shape: " + str(X_train.shape))

print ("Val_shape: " + str(X_val.shape))

print ("No of positives in train: " + str(Y_train.sum()))

print ("No of positives in val: " + str(Y_val.sum()))
lgb_train = lgb.Dataset(X_train, Y_train)

lgb_eval = lgb.Dataset(X_val, Y_val)



params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 12,

    'learning_rate': 0.001,

    'feature_fraction': 0.7,

    'bagging_fraction': 0.7,

    'bagging_freq': 5,

}



print('Start training...')



gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_eval], 

                early_stopping_rounds=100, verbose_eval=25)
from sklearn.metrics import accuracy_score

# training metrics



pred_train = gbm.predict(train_df[cols], num_iteration=gbm.best_iteration)

pred_train = np.where(pred_train > 0.5, 1, 0)

print('Accurracy: {}'.format(accuracy_score(train_df['label1'], pred_train)))
from sklearn.metrics import confusion_matrix, recall_score, precision_score



print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')

cm = confusion_matrix(train_df['label1'], pred_train)

print(cm)
pred_test = gbm.predict(test_df[cols], num_iteration=gbm.best_iteration)

pred_test = np.where(pred_test > 0.5, 1, 0)

print('Accurracy: {}'.format(accuracy_score(test_df['label1'], pred_test)))
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')

cm = confusion_matrix(test_df['label1'], pred_test)

print(cm)
# compute precision and recall

precision_test = precision_score(test_df['label1'], pred_test)

recall_test = recall_score(test_df['label1'], pred_test)

f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )