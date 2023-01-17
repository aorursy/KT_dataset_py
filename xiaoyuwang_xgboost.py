import xgboost as xgb
import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['wkday'] = train.time // 1440 % 7
train['dtime'] = train.time % 1440
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train['place_id'])
X = train[['x', 'y', 'accuracy', 'wkday', 'dtime']]
param = {'bst:max_depth':10, 'bst:eta':1, 'silent':1, 'objective':'multi:softmax', 'eval_metric':'auc' }
goal = xgb.DMatrix(X, label = y)

