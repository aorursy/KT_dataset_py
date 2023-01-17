!pip install --use-feature=2020-resolver https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2B00b0ad1293b4fa74d6aca5da4e9ab7a9d16777f0-py3-none-manylinux2010_x86_64.whl
import xgboost as xgb

xgb.__version__
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')

test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv.zip')



for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        

for c in test.columns:

    if test[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(test[c].values)) 

        test[c] = lbl.transform(list(test[c].values))

        

y_train = train["y"]

train.drop('y', axis=1, inplace=True)





xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.7,

    'objective': 'reg:squarederror',

    'eval_metric': 'rmse',

    'silent': 1,

    'tree_method': 'gpu_hist', 

    'predictor': 'gpu_predictor'

}



dtrain = xgb.DMatrix(train, y_train)

dtest = xgb.DMatrix(test)
train.shape
test.shape
%%time

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=500)
%%time

model.predict(dtest, pred_contribs=True)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.7,

    'objective': 'reg:squarederror',

    'eval_metric': 'rmse',

    'silent': 1,

    'tree_method': 'hist', 

    'predictor': 'cpu_predictor'

}
%%time

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=500)
%%time

model.predict(dtest, pred_contribs=True)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.8,

    'colsample_bytree': 0.7,

    'objective': 'reg:squarederror',

    'eval_metric': 'rmse',

    'silent': 1,

    'tree_method': 'exact', 

    'predictor': 'cpu_predictor'

}
%%time

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=500)
%%time

model.predict(dtest, pred_contribs=True)