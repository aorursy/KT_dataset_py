import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = train[train.columns[2:]]

test = test[test.columns[1:]]
features = train.columns

train['target'] = 0

test['target'] = 1
features
train_test = pd.concat([train, test], axis =0)



target = train_test['target'].values
param = {'num_leaves': 40,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 4,

         'learning_rate': 0.006,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 27,

         "metric": 'auc',

         "verbosity": -1}



folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof = np.zeros(len(train_test))





for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train_test.iloc[trn_idx][features], label=target[trn_idx])

    val_data = lgb.Dataset(train_test.iloc[val_idx][features], label=target[val_idx])



    num_round = 30000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1400)

    oof[val_idx] = clf.predict(train_test.iloc[val_idx][features], num_iteration=clf.best_iteration)
metrics.roc_auc_score(target, oof)