import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import shap

import math

import os

print(os.listdir("../input"))

from sklearn import preprocessing

import xgboost as xgb

import gc





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
train_time = train['time'].values

train_time_0 = train_time[:50000]

for i in range(1,100):

    train_time_0 = np.hstack([train_time_0, train_time[:50000]])

train['time'] = train_time_0





train_time_1 = train_time[:50000]

for i in range(1,40):

    train_time_1 = np.hstack([train_time_1, train_time[:50000]])

test['time'] = train_time_1
np.unique(train_time_1).shape
np.unique(train_time_0).shape
n_groups = 100

train["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    train.loc[ids,"group"] = i

    

n_groups = 40

test["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    test.loc[ids,"group"] = i

    

train['signal_scaled'] = 0

test['signal_scaled'] = 0

train['signal_2'] = 0

test['signal_2'] = 0
n_groups = 100

for i in range(n_groups):

    sub = train[train.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    train.loc[sub.index,"signal_scaled"] = list(np.array(signals))

    train.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))
n_groups = 40

for i in range(n_groups):

    sub = test[test.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    test.loc[sub.index,"signal_scaled"] = list(np.array(signals))

    test.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))
train.head()
test.head()
del train['open_channels']

del train['group']



train.head()
test.head()
del test['group']

test.head()
train['target'] = 0

test['target'] = 1
train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values





del train, test

gc.collect()
train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)

del train_test

gc.collect()
train_y = train['target'].values

test_y = test['target'].values

del train['target'], test['target']

gc.collect()
train = lgb.Dataset(train, label=train_y)

test = lgb.Dataset(test, label=test_y)

gc.collect()
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 2,

         'learning_rate': 0.2,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}
num_round = 50

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)
columns = ['time', 'signal', 'signal_scaled', 'signal_2']
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),columns)), columns=['Value','Feature'])



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')