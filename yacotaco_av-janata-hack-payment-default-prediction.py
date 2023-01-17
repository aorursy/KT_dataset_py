# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from scipy.stats import lognorm, gamma

import collections



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/train_20D8GL3.csv')

test = pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/test_O6kKpvt.csv')

sample_sub = pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/sample_submission_gm6gE0l.csv')
train.shape
test.shape
train.head()
train.describe()
train.info()
print(" DUPLICATED: ", len(train[train.duplicated()]), "\n", "MISSING: ", train.isnull().sum().sum())
test.head()
sample_sub.head()
# Target Variable: Default payment (1=yes, 0=no)

plt.hist(train['default_payment_next_month'])

plt.title('Target Variable')

plt.show()
# LIMIT_BAL Amount of given credit (NT dollars)

ax = sns.distplot(train['LIMIT_BAL'])

ax.autoscale()

plt.show()
ax = sns.distplot(train['AGE'])

ax.autoscale()

plt.show()
# SEX Gender (1=male, 2=female)

plt.hist(train['SEX'])

plt.title('SEX')

plt.show()
# EDUCATION (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

plt.hist(train['EDUCATION'])

plt.title('EDUCATION')

plt.show()
import seaborn as sns; sns.set()

plt.figure(figsize=(25, 25))

ax = sns.heatmap(train.corr(), vmin=-1, vmax=1, center=0, annot=True, square=True)
# train['LOG_LIMIT_BAL'] = np.log(train['LIMIT_BAL'])

# test['LOG_LIMIT_BAL'] = np.log(test['LIMIT_BAL'])



X_train = train.drop(['default_payment_next_month', 'ID'], axis=1)

y_train = train['default_payment_next_month'].values



X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)



X_test = test.drop(['ID'], axis=1)



lgb_train = lgb.Dataset(X_tr, y_tr)

lgb_val = lgb.Dataset(X_val, y_val)
evals_result = {} 



params = {

        'objective': 'binary',

        'boosting': 'gbdt',

        'learning_rate': 0.03,

        'num_leaves': 100,

        'bagging_fraction': 0.95,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 0.9,

        'feature_fraction_seed': 1,

        'max_bin': 256,

        'metric' : 'auc',

        'n_estimators': 250,

    }



cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=5, metrics='auc', early_stopping_rounds=100, seed=50)



lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, evals_result=evals_result, verbose_eval=5)
# plot cv metric

od = collections.OrderedDict()

d = {}

results = cv_results['auc-mean']

od['auc'] = results

d['cv'] = od



ax = lgb.plot_metric(d,title='Metric during cross-validation', metric='auc')

plt.show()



print("CV best score: " + str(max(cv_results['auc-mean'])))
# plot train metric 

ax = lgb.plot_metric(evals_result, metric='auc')

plt.show()



print("Train best score: " + str(max(evals_result['valid_0']['auc'])))



# plot feature importance

lgb.plot_importance(lgbm_model)



# plot tree

# lgb.plot_tree(lgbm_model,  figsize=(50, 50))
predictions = lgbm_model.predict(X_test)



# Writing output to file

subm = pd.DataFrame()

subm['ID'] = test['ID']

subm['default_payment_next_month'] = predictions



subm.to_csv("/kaggle/working/" + 'submission.csv', index=False)

subm