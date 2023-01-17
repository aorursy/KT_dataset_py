# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sklearn.metrics

import numpy as np

import pandas as pd

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

import warnings

import time

import os

from sklearn.model_selection import train_test_split



warnings.filterwarnings('ignore')

np.random.seed(4590)



print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/data-cleaning-based-3691/clean_test_data_withholidays.csv')

df_train = pd.read_csv('../input/data-cleaning-based-3691/clean_train_data_withholidays.csv')
df_train_columns = [c for c in df_train.columns if c not in ['first_active_month', 'target', 'card_id', 'outliers',

                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',

                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',

                  'OOF_PRED', 'month_0']]

target = df_train['target']

del df_train['target']
# Optuma method to get the best value from commite 8 

param = {'num_leaves': 668,

         'min_data_in_leaf': 94, 

         'objective':'regression',

         'max_depth': 14,

         'learning_rate': 0.018,

        # "min_child_samples": 20,

         "boosting": "rf",

         "feature_fraction": 0.89,

         'reg_alpha': 5,

         'reg_lambda': 4,

         'min_split_gain':9,

         "bagging_fraction":0.71,

         "bagging_freq": 2,

         'min_child_weight': 47,

         "metric": 'rmse',

         "verbosity": -1,

         "nthread": 4,

         "random_state": 6756}

folds = StratifiedKFold(n_splits=7, shuffle=True, random_state=4590)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = df_train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))

pd.DataFrame(oof).to_csv("train_predict_dart.csv")
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission_dart.csv", index=False)