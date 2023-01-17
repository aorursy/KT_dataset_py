# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from scipy.stats import zscore

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

import pandas as pd

import io

import requests

import numpy as np

from sklearn import metrics

from sklearn.model_selection import KFold

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

import lightgbm as lgb

from bayes_opt import BayesianOptimization

import warnings, gc

# Any results you write to the current directory are saved as output.
def multi_merge(left,right,*args):

    start = args[0]

    for i in range(1,len(args)):

        start = start.merge(args[i], how = 'left', left_on = left, right_on = right)

    return start



def to_xy(df, target):

    result = []

    for x in df.columns:

        if x != target:

            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(

    target_type = df[target].dtypes

    target_type = target_type[0] if hasattr(

        target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.

    if target_type not in (np.int64, np.int32):

        # Classification

        dummies = pd.get_dummies(df[target])

        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)

    # Regression

    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)
df_train = pd.read_csv('../input/midterm/train.csv')

df_test = pd.read_csv('../input/midterm/test.csv')

df_detail = pd.read_csv('../input/2016-soi-tax-stats/16zpallagi.csv')

df_detail = df_detail.loc[(df_detail['zipcode'] != 0) & (df_detail['zipcode'] != 99999)].reset_index(drop=True)
import re

column_list = list(df_detail.columns)

A_list = [column for column in column_list if re.match(r'^[A][0-9]{2,10}',column)]

N_list = [column for column in column_list if re.match(r'^[N][0-9]{2,10}',column)]

O_list = [column for column in column_list if (column not in A_list and column not in N_list)]
special = []

for name in A_list:

    if 'N' + name[1:] not in N_list:

        special.append(name)

df_zipcode = df_detail.loc[:,'zipcode']

df_detail_pair = df_detail.iloc[:,18:].drop(special+['SCHF'],axis=1)

df_detail_pair = pd.concat([df_zipcode,df_detail_pair],axis = 1)

for name in N_list:

    A_name = 'A' + name[1:]

    df_detail_pair[A_name] = df_detail_pair[A_name]*df_detail_pair[name]
df_detail_pair = df_detail_pair.groupby('zipcode').sum()

for name in N_list:

    A_name = 'A' + name[1:]

    df_detail_pair[A_name] = df_detail_pair[A_name]/df_detail_pair[name]

df_feature_1 = df_detail_pair.loc[:,['A' + name[1:] for name in N_list]].fillna(0)
df_special_feature_1 = df_detail.loc[:,special+['zipcode','N1']]

df_special_feature_1 = df_special_feature_1.groupby(by='zipcode').apply(lambda x: pd.Series({'avg_agi':sum(x['N1']*x[special[0]])/sum(x['N1']),'avg_item_r':sum(x['N1']*x[special[1]])/sum(x['N1'])}))
df_special_feature_2 = df_detail.iloc[:,:18]

for column in list(df_special_feature_2.columns)[5:]:

    df_special_feature_2[column] = df_special_feature_2[column]/df_special_feature_2['N1']

df_special_feature_2 = df_special_feature_2.fillna(0).drop(['STATEFIPS','STATE','agi_stub'],axis=1).groupby(by='zipcode').agg('mean')
df_full_detail = multi_merge('zipcode','zipcode',df_special_feature_2,df_special_feature_1,df_feature_1)

df_full_detail.head()
for name in df_full_detail.columns:

    if df_full_detail[name].dtype in ('float64','int64'):

        df_full_detail[name] = zscore(df_full_detail[name])

df_train = multi_merge('zipcode','zipcode',df_train,df_full_detail).drop(['id','zipcode'],axis=1)

df_test = multi_merge('zipcode','zipcode',df_test,df_full_detail).drop(['id','zipcode'],axis=1)

x,y = to_xy(df_train,'score')
import lightgbm as lgb

from bayes_opt import BayesianOptimization

import warnings, gc

features = [c for c in df_train.columns if c not in ['score']]

df_target = df_train['score']

def lgb_cv(max_depth,

          num_leaves,

          min_data_in_leaf,

          feature_fraction,

          bagging_fraction,

          lambda_l1):

    folds = KFold(n_splits=5, shuffle=True, random_state=15)

    oof = np.zeros(df_train.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_target.values)):

        print("fold n°{}".format(fold_))

        trn_data = lgb.Dataset(df_train.iloc[trn_idx][features],

                               label=df_target.iloc[trn_idx]

                              )

        val_data = lgb.Dataset(df_train.iloc[val_idx][features],

                               label=df_target.iloc[val_idx]

                              )

        param = {

            'num_leaves': int(num_leaves),

            'min_data_in_leaf': int(min_data_in_leaf), 

            'objective':'regression',

            'max_depth': int(max_depth),

            'learning_rate': 0.005,

            "boosting": "gbdt",

            "feature_fraction": feature_fraction,

            "bagging_freq": 1,

            "bagging_fraction": bagging_fraction ,

            "bagging_seed": 11,

            "metric": 'rmse',

            "lambda_l1": lambda_l1,

            "verbosity": -1

        }

        clf = lgb.train(param,

                        trn_data,

                        10000,

                        valid_sets = [trn_data, val_data],

                        verbose_eval=200,

                        early_stopping_rounds = 100)

        oof[val_idx] = clf.predict(df_train.iloc[val_idx][features],

                                   num_iteration=clf.best_iteration)

        del clf, trn_idx, val_idx

        gc.collect()

    return -metrics.mean_squared_error(oof, df_target)**0.5
LGB_BO = BayesianOptimization(lgb_cv, {

    'max_depth': (4, 10),

    'num_leaves': (5, 130),

    'min_data_in_leaf': (10, 150),

    'feature_fraction': (0.7, 1.0),

    'bagging_fraction': (0.7, 1.0),

    'lambda_l1': (0, 6)

    })
print('-'*126)

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=2, n_iter=4, acq='ei', xi=0.0)
features = [c for c in df_train.columns if c not in ['score']]

df_target = df_train['score']

param = {

            'num_leaves': int(52),

            'min_data_in_leaf': int(12), 

            'objective':'regression',

            'max_depth': int(6),

            'learning_rate': 0.005,

            "boosting": "gbdt",

            "feature_fraction": 0.7066,

            "bagging_freq": 1,

            "bagging_fraction": 0.8086,

            "bagging_seed": 11,

            "metric": 'rmse',

            "lambda_l1": 3.651,

            "random_state": 11,

            "verbosity": -1

        }

folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, df_target.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features],

                           label=df_target.iloc[trn_idx]

                          )

    val_data = lgb.Dataset(df_train.iloc[val_idx][features],

                           label=df_target.iloc[val_idx]

                          )



    num_round = 10000

    clf = lgb.train(param,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=100,

                    early_stopping_rounds = 200)

    

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(metrics.mean_squared_error(oof, df_target)**0.5))
submit_csv = pd.read_csv('../input/midterm/test.csv')

submit_csv = submit_csv.drop('zipcode',axis=1)

submit_csv['score'] = predictions

submit_csv.to_csv('csv_to_submit_80.csv', index = False)
import matplotlib.pyplot as plt

import seaborn as sns

cols = (feature_importance_df[["feature", "importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
#features = list(cols)[:10]

#features_train = features + ['score']

#df_train = df_train.loc[:,features_train]

#df_test = df_test.loc[:,features]

#x,y = to_xy(df_train,'score')



kf = KFold(5)

oos_y = []

oos_pred = []

fold = 0

for train,test in kf.split(x):

    fold += 1

    print('Fold #{}'.format(fold))

    x_train = x[train]

    y_train = y[train]

    x_test = x[test]

    y_test = y[test]

    

    model = Sequential()

    model.add(Dense(100, input_dim=x.shape[1], activation='relu'))

    model.add(Dense(75, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=10, verbose=1, mode='auto')

    checkpointer = ModelCheckpoint(filepath="midterm_model1_best", verbose=0, save_best_only=True)

    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=0,epochs=250)

    

    pred = model.predict(x_test)

    oos_y.append(y_test)

    oos_pred.append(pred)

    score = np.sqrt(metrics.mean_squared_error(pred,y_test))

    print("Fold score (RMSE): {}".format(score))



oos_y = np.concatenate(oos_y)

oos_pred = np.concatenate(oos_pred)

score = metrics.mean_squared_error(oos_pred,oos_y)

print("Final score (RMSE): {}".format(np.sqrt(score)))



df_test = pd.read_csv('../input/midterm/test.csv')

df_test = multi_merge('zipcode','zipcode',df_test,df_full_detail).drop(['id','zipcode'],axis=1)

true_test = df_test.values.astype(np.float32)

pred_test = model.predict(true_test)

pred_test = model.predict(true_test)

final_test_score = np.concatenate(pred_test)



df_test['score'] = final_test_score

df_test = df_test.loc[:,['id','score']]

df_test.to_csv('csv_to_submit_80_neural.csv', index = False)
