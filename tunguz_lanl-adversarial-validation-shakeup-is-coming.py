# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/andrews-features-only/X_tr.csv')

test = pd.read_csv('../input/andrews-features-only/X_test.csv')
train.shape
test.shape
features = train.columns

train['target'] = 0

test['target'] = 1
train_test = pd.concat([train, test], axis =0)



target = train_test['target'].values
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

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
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),train.columns)), columns=['Value','Feature'])



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')