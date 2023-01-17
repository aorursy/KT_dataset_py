# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics



from sklearn import preprocessing

import gc





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





# Load data





test = pd.read_csv('../input/best-features-only/X_test.csv')

features = test.columns

train = pd.read_csv('../input/best-features-only/X_train.csv', usecols=features)



trs = train.shape[0]

tes = test.shape[0]



print(features)

print(train.shape)

print(test.shape)
train = pd.concat([train, test], axis =0)

del test

gc.collect()
target = np.hstack([np.zeros(trs,), np.ones(tes,)])

train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=42, shuffle=True)

del target

gc.collect()
train = lgb.Dataset(train, label=y_train)

test = lgb.Dataset(test, label=y_test)
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 56,

         "metric": 'auc',

         "verbosity": -1}
num_round = 50

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')
%%time

del train, test, clf



gc.collect()

gc.collect()



features = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'month',

       'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',

       'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_7', 'lag_28',

       'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28',

       'quarter', 'mday']



test = pd.read_csv('../input/best-features-only/X_test.csv', usecols=features)

train = pd.read_csv('../input/best-features-only/X_train.csv', usecols=features)



train = pd.concat([train, test], axis =0)

del test

gc.collect()



target = np.hstack([np.zeros(trs,), np.ones(tes,)])



train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=42, shuffle=True)

del target

gc.collect()



train = lgb.Dataset(train, label=y_train)

test = lgb.Dataset(test, label=y_test)



clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)



feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-02.png')
%%time

del train, test, clf



gc.collect()

gc.collect()



features = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',

       'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',

       'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_7', 'lag_28',

       'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28',

       'quarter', 'mday']



test = pd.read_csv('../input/best-features-only/X_test.csv', usecols=features)

train = pd.read_csv('../input/best-features-only/X_train.csv', usecols=features)



train = pd.concat([train, test], axis =0)

del test

gc.collect()



target = np.hstack([np.zeros(trs,), np.ones(tes,)])



train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=42, shuffle=True)

del target

gc.collect()



train = lgb.Dataset(train, label=y_train)

test = lgb.Dataset(test, label=y_test)



clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)



feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-03.png')
%%time

del train, test, clf



gc.collect()

gc.collect()



features = ['dept_id', 'store_id', 'cat_id', 'state_id', 'wday',

       'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',

       'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_7', 'lag_28',

       'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28',

       'quarter', 'mday']



test = pd.read_csv('../input/best-features-only/X_test.csv', usecols=features)

train = pd.read_csv('../input/best-features-only/X_train.csv', usecols=features)



train = pd.concat([train, test], axis =0)

del test

gc.collect()



target = np.hstack([np.zeros(trs,), np.ones(tes,)])



train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=42, shuffle=True)

del target

gc.collect()



train = lgb.Dataset(train, label=y_train)

test = lgb.Dataset(test, label=y_test)



clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)



feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-04.png')