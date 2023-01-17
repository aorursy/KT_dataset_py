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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
import lightgbm as lgb

from sklearn.model_selection import cross_validate, StratifiedKFold

from sklearn.metrics import roc_auc_score
feature = train_df.columns[1:-1]

target = train_df.columns[-1:]
X = train_df[feature]

y = train_df[target]

X_test = test_df[feature]
all_K_fold_results = []

kf = StratifiedKFold(n_splits=5, shuffle = True)

oof_preds = np.zeros(X.shape[0])

sub_preds = np.zeros(test_df.shape[0])

params = {

    'metric' : 'auc'

}



for fold_, (trn_index, val_index) in enumerate(kf.split(X,y)):

    X_trn, X_val = X.iloc[trn_index], X.iloc[val_index]

    y_trn, y_val = y.iloc[trn_index], y.iloc[val_index]

    

    print("Fold:{}".format(fold_ + 1))

    

    trn_data = lgb.Dataset(X_trn, label = y_trn)

    val_data = lgb.Dataset(X_val, label = y_val)

    

    clf = lgb.train(params, trn_data, 500, valid_sets = [trn_data, val_data]

                   , verbose_eval = 100, early_stopping_rounds = 50)

    

    oof_preds[val_index] = clf.predict(X_val, num_iteration = clf.best_iteration)

    

    sub_preds += clf.predict(X_test, num_iteration = clf.best_iteration) / kf.n_splits

    

print("CV score : {:<8.5f}".format(roc_auc_score(y, oof_preds)))
sub_preds
submission = pd.read_csv("../input/sample_submission.csv")

submission[target] = sub_preds
submission.to_csv('basic_start.csv', index=False)