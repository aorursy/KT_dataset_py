import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

import lightgbm as lgb

import gc

import matplotlib.pyplot as plt

import  warnings

warnings.simplefilter('ignore')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train_x=train.drop(['ID_code','target'],axis=1)

corr_matrix = train_x.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.995)]
#len(to_drop)=0

from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

#train_x=train.drop(['card_id','target'],axis=1)

#train_x=train_x.drop(to_drop,axis=1)

target=train.target

train_id=train.ID_code

test_id=test.ID_code

test_df=test.drop('ID_code',axis=1)

test_df=test_df.drop(to_drop,axis=1)
#target.sum()/(len(train)-target.sum())=0.1117163789174106

param = {'num_leaves': 21,

         'min_data_in_leaf': 35, 

         'objective':'binary',

         'max_depth': -1,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.6,

#         "bagging_freq": 1,

         "bagging_fraction": 0.8,

         "bagging_seed": 42,

         "metric": 'auc',

#         "lambda_l1": 160,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2019}
train_columns=train_x.columns

evescore=pd.DataFrame({'training':[0],'valid_1':[0]},index=['rmse'])

#train_columns=list(train_x.columns)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x,target)): # fold_ is the indicesnof split!

    trn_data = lgb.Dataset(train_x.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(train_x.iloc[val_idx][train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 100000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000,early_stopping_rounds = 300)

    oof[val_idx] = clf.predict(train_x.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    evescore=evescore+pd.DataFrame(clf.best_score)

    #predictions

    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) 
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
predictions/5

lvl_one_train=pd.DataFrame(train_id)

lvl_one_train['target']=oof

lvl_one_test=pd.DataFrame(test_id)

lvl_one_test['target']=predictions

lvl_one_train.to_csv('lgb_train1.csv',index=False)

lvl_one_test.to_csv('lgb_test1.csv',index=False)