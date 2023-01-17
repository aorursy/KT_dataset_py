# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data.info()
data.dtypes
data.isnull().values.any()
test.isnull().values.any()
data.describe()
x = data.iloc[:,2:]
y = data.iloc[:,1]
features = x.columns
param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.32,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.045,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'lambda_l2':0.1,
        'objective': 'binary', 
        'verbosity': 1
    }
num_round = 1000000
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=32)
oof = np.zeros(len(data))
predictions = np.zeros(len(test))
for fold_no, (trn_idx, val_idx) in enumerate(folds.split(x.values, y.values)):
    print("Fold {}".format(fold_no))
    trn_data = lgb.Dataset(x.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(x.iloc[val_idx], label=y.iloc[val_idx])
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3500)
    oof[val_idx] = clf.predict(x.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))
print(oof)
print(predictions)
sub = pd.DataFrame()
sub['ID_code'] = test.iloc[:,0]
sub['pred'] = pd.Series(np.array(predictions))
sub.head()
test.head()
sub.to_csv('submission.csv')