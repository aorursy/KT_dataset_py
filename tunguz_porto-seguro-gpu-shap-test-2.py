!pip install --use-feature=2020-resolver https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2B00b0ad1293b4fa74d6aca5da4e9ab7a9d16777f0-py3-none-manylinux2010_x86_64.whl
import xgboost as xgb

xgb.__version__
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/porto-seguro-safe-driver-prediction"))
train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')
train.shape
test.shape
y_train = train["target"]

train.drop('target', axis=1, inplace=True)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.7,

    'objective': 'reg:logistic',

    'eval_metric': 'auc',

    'verbosity': 0,

    'tree_method': 'gpu_hist', 

    'predictor': 'gpu_predictor'

}



dtrain = xgb.DMatrix(train, y_train)

dtest = xgb.DMatrix(test)
del train, test

gc.collect()
gc.collect()

gc.collect()
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=500)
%%time

model.predict(dtest, pred_contribs=True)
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=1000)
%%time

model.predict(dtest, pred_contribs=True)
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=1500)
%%time

model.predict(dtest, pred_contribs=True)
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=2000)
%%time

model.predict(dtest, pred_contribs=True)
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=2500)
%%time

model.predict(dtest, pred_contribs=True)
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=3000)
%%time

model.predict(dtest, pred_contribs=True)