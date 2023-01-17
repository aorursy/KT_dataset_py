import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import shap

import os

print(os.listdir("../input"))

from sklearn import preprocessing

import xgboost as xgb

import gc





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
%%time



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
features = test.columns

train['target'] = 0

test['target'] = 1

train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values

del train, test

gc.collect()
# Label Encoding

for f in features:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train_test[f].values) )

    train_test[f] = lbl.transform(list(train_test[f].values))
train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)

del train_test

gc.collect()
train_y = train['target'].values

test_y = test['target'].values

del train['target'], test['target']

gc.collect()
train = lgb.Dataset(train, label=train_y)

test = lgb.Dataset(test, label=test_y)
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.001,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}
num_round = 50000

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 500)