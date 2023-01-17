import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split

import seaborn as sns

import shap

import matplotlib.pyplot as plt

%matplotlib inline

pd.options.display.max_columns=999
# load data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

(df_train.shape, df_test.shape)
df_train.drop('ID_code', axis=1, inplace=True)

df_test.drop('ID_code', axis=1, inplace=True)
var_cols = df_train.columns.drop('target')
params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 13,

    'learning_rate': 0.01,

    'feature_fraction': 0.1,

    'bagging_fraction': 0.3,

    'bagging_freq': 5,

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf':10.0,

    'num_boost_round':999999,

    'early_stopping_rounds':500,

    'random_state':2019

}
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

df_train[var_cols] = mms.fit_transform(df_train[var_cols])

df_test[var_cols] = mms.fit_transform(df_test[var_cols])
X, y = df_train.drop(['target'], axis=1), df_train.target.values



X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=2019)



lgb_trn = lgb.Dataset(X_trn, y_trn)

lgb_val = lgb.Dataset(X_val, y_val)

model = lgb.train(params, lgb_trn, valid_sets=[lgb_trn, lgb_val], valid_names=['train','valid'],verbose_eval=2000)
p = model.predict(df_test)

sub = pd.read_csv('../input/sample_submission.csv')

sub.target = p

sub.to_csv('sub.csv', index=False)
# plot feature importance

feature_importance = pd.DataFrame(columns=['feature','importance'])

feature_importance.feature = X.columns.values

feature_importance.importance = model.feature_importance()

feature_importance.sort_values(by='importance', ascending=False, inplace=True)



plt.figure(figsize=(10,50))

sns.barplot('importance', 'feature', data=feature_importance)
# shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)
# summarize the effects of all the features

shap.summary_plot(shap_values, X)
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(50,50))

for i in range(10):

    for j in range(10):

        ids = i*10+j

        sns.scatterplot(X['var_'+str(ids)], shap_values[:,ids], ax=ax[i,j])
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(50,50))

for i in range(10,20):

    for j in range(10):

        ids = i*10+j

        sns.scatterplot(X['var_'+str(ids)], shap_values[:,ids], ax=ax[i-10,j])