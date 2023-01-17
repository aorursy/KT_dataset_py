# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def activation(x):
    return np.log1p(x)
def deactivation(x):
    return np.expm1(x)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
_x_train = df_train.drop(['ID', 'target'], axis=1).astype('float32')
_x_test = df_test.drop(['ID'], axis=1).astype('float32')
y_train = activation(df_train['target'])
del df_train
del df_test
gc.collect()
n_dim = 32
svd_model = TruncatedSVD(n_dim, random_state=0)
svd_data_train = pd.DataFrame(svd_model.fit_transform(_x_train), columns=[f'svd_{i}' for i in range(n_dim)]).astype('float32')
svd_data_test = pd.DataFrame(svd_model.transform(_x_test), columns=[f'svd_{i}' for i in range(n_dim)]).astype('float32')
"""
df_cnt_train = pd.DataFrame()
df_cnt_test = pd.DataFrame()

for col in tqdm(_x_train.columns.values):
    if len(_x_train[col].unique()) > 100:
        continue
    tmp = _x_train.groupby(col)[col].count()
    tmp.index.name = 'aaa'
    tmp = tmp.reset_index()
    tmp.columns = [col, col + '_cnt']
    ret = pd.merge(_x_train[[col]], tmp, how='left', on=col)
    df_cnt_train[col + '_cnt'] = ret[col + '_cnt'].values
    ret = pd.merge(_x_test[[col]], tmp, how='left', on=col)
    df_cnt_test[col + '_cnt'] = ret[col + '_cnt'].values
    #x_test = pd.merge(x_test, tmp, how='left', on=col)
    df_cnt_train = df_cnt_train.astype('float32')
df_cnt_test = df_cnt_test.astype('float32')
"""
gc.collect()
x_train = pd.concat([_x_train,
                     svd_data_train,
                     #df_cnt_train
                    ], axis=1, copy=False)
x_test = pd.concat([_x_test,
                    svd_data_test, 
                    #df_cnt_test
                   ], axis=1, copy=False)
#del svd_data_train
#del df_cnt_train
#del svd_data_test
#del df_cnt_test
gc.collect()
all_params = {'min_child_weight': [3],
                  'subsample': [1],
                  'subsample_freq': [1],
                  'seed': [114514],
                  'colsample_bytree': [0.8],
                  'learning_rate': [0.01],
                  'max_depth': [-1],
                  'min_split_gain': [0.01],
                  'reg_alpha': [0],
                  'max_bin': [63],
                  'num_leaves': [255],
                  'objective': ['regression_l1'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  'boosting_type': ['gbdt'],
                  'metric': ['rmse'],
                  }

min_score = 10000
min_params = None
cv = KFold(n_splits=5, shuffle=True, random_state=114514)  
for params in tqdm(list(ParameterGrid(all_params))):
    for test, train in cv.split(x_train, y_train):
        trn_x = x_train.iloc[train]
        val_x = x_train.iloc[test]
        trn_y = y_train[train]
        val_y = y_train[test]
        train_data = lgb.Dataset(trn_x,
                                 label=trn_y
                                 )
        test_data = lgb.Dataset(val_x,
                                label=val_y
                                )
        clf = lgb.train(params,
                        train_data,
                        100000,  # params['n_estimators'],
                        early_stopping_rounds=100,
                        valid_sets=[test_data],
                        verbose_eval=30
                       )
        params['n_tree'] = clf.best_iteration
        pred = clf.predict(val_x)
        score = np.sqrt(mean_squared_error(val_y, pred))
        if min_score > score:                                                                                                                                                  
            min_score = score                                                                                                                                                                        
            min_params = params 
        print(score, min_score)
        break


print(min_score, min_params)
train_data = lgb.Dataset(x_train,
                             label=y_train,   
                             )
n_tree = int(clf.best_iteration * 1.1)
clf = lgb.train(min_params,
                    train_data,
                    n_tree,
                    valid_sets=[train_data],
                    verbose_eval=10
                    )
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = deactivation(clf.predict(x_test))
sub.to_csv('submit.csv', index=False)