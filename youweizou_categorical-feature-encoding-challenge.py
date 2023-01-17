# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

import scipy

from sklearn.preprocessing import StandardScaler, normalize

import lightgbm as lgb

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from hyperopt import tpe, fmin, Trials, hp

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier, Pool
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

train.shape, test.shape
dataset = pd.concat([train, test]).drop(columns='id')
# plt.figure(figsize=(15, 20))

# i = 0

# for c in dataset:

#     plt.subplot(5, 4, i+1)

#     sns.countplot(dataset[c])

#     i += 1

# plt.tight_layout()
map_ord1 = {'Novice':1, 

            'Contributor':2, 

            'Expert':4, 

            'Master':5, 

            'Grandmaster':6}

dataset['ord_1'] = dataset['ord_1'].map(map_ord1, na_action='ignore')
map_ord2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

dataset['ord_2'] = dataset['ord_2'].map(map_ord2, na_action='ignore')
for i in range(3, 6):

    col = 'ord_' + str(i)

    sorted_cat = sorted(list(dataset[col].dropna().unique()))

    dict_ = {}

    val = 0

    for e in sorted_cat:

        dict_[e] = val

        val += 1

    dataset[col] = dataset[col].map(dict_, na_action='ignore')
num_col = ['ord_'+str(i) for i in range(6)] + ['day', 'month']

cat_data = dataset.drop(columns=num_col+['target'])

# num_data = dataset[num_col]

for c in cat_data.columns:

    if cat_data[c].dtype != 'object':

        cat_data[c] = cat_data[c].astype('object')
cat_dummies = pd.get_dummies(cat_data, dummy_na=True, sparse=True).sparse.to_coo()
# plt.figure(figsize=(15, 20))

# i = 0

# for c in dataset:

#     plt.subplot(5, 4, i+1)

#     sns.countplot(dataset[c])

#     i += 1

# plt.tight_layout()
# for i in range(3):

#     col = 'bin_' + str(i)

#     dataset[col] = dataset[col].astype('object')

# dataset['day'] = dataset['day'].astype('object')

# dataset['month'] = dataset['month'].astype('object')

# dataset = pd.get_dummies(dataset)
imputer = SimpleImputer(strategy='mean')

num_data = pd.DataFrame(imputer.fit_transform(dataset[num_col]), columns=num_col)

data =scipy.sparse.hstack([cat_dummies, num_data]).tocsr()
df_train = data[:len(train)]

df_test = data[len(train):]

y = dataset['target'].dropna()

# df_train
scale = StandardScaler(with_mean=False)

train_value = scale.fit_transform(df_train)

test_value = scale.transform(df_test)
x_train, x_valid, y_train, y_valid = train_test_split(train_value, y, test_size=0.3, random_state=80000)

train_data = lgb.Dataset(x_train, y_train)

valid_data = lgb.Dataset(x_valid, y_valid)

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'application': 'binary',

    'is_unbalance': True,

    'num_leaves': 30,

    'learning_rate': 0.01,

    'bagging_fraction': 0.3, 

    'bagging_freq': 1,

    'feature_fraction': 0.3, 

    'metric':'auc'

}

model = lgb.train(params, train_data, valid_sets=valid_data, early_stopping_rounds=50, verbose_eval=200, num_boost_round=5000)
def f(params):

    model = lgb.train(params, train_data, valid_sets=valid_data, early_stopping_rounds=50, verbose_eval=200, num_boost_round=5000)

    return -model.best_score['valid_0']['auc']



space = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'application': 'binary',

    'is_unbalance': True,

    'num_leaves': hp.choice('num_leaves', range(20, 41, 10)),

    'learning_rate': 0.01,

    'bagging_fraction': 0.3,

    'bagging_freq': hp.choice('bagging_freq', range(1, 4)),

    'feature_fraction': hp.choice('feature_fraction', [0.2, 0.3, 0.4, 0.5, 0.6]),

    'metric':'auc'

}
# trial = Trials()

# best = fmin(f, space=space, algo=tpe.suggest, trials=trial, max_evals=40)
sub = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

sub['target'] = model.predict(test_value)

sub.to_csv('sub.csv', index=False)
sub