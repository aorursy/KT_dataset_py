import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numba import jit

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import pyplot

from sklearn.cluster import KMeans

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Descriptive statistics train

# shape

print(train.shape)

# types

print(train.dtypes)

# info

print(train.info())

# head

print(train.head(10))

# descriptions, change precision to 2 places

set_option('precision', 1)

train.describe()
# Descriptive statistics test

# shape

print(test.shape)

# types

print(test.dtypes)

# info

print(test.info())

# head

print(test.head(10))

# descriptions, change precision to 2 places

set_option('precision', 1)

test.describe()
# Analyzing missing values train

print(train.isnull().sum())
# Analyzing missing values test

print(test.isnull().sum())
#Data visualization

sns.countplot(train['target'])
train.iloc[:,2:10].hist()

train.iloc[:,11:20].hist()

train.iloc[:,21:30].hist()

train.iloc[:,31:40].hist()

train.iloc[:,41:50].hist()

train.iloc[:,51:60].hist()

train.iloc[:,61:70].hist()

train.iloc[:,71:80].hist()

train.iloc[:,81:90].hist()

train.iloc[:,101:110].hist()

train.iloc[:,111:120].hist()

train.iloc[:,121:130].hist()

train.iloc[:,131:140].hist()

train.iloc[:,141:150].hist()

train.iloc[:,151:160].hist()

train.iloc[:,161:170].hist()

train.iloc[:,171:180].hist()

train.iloc[:,181:190].hist()

train.iloc[:,191:200].hist()

pyplot.show()
test.iloc[:,1:10].hist()

test.iloc[:,11:20].hist()

test.iloc[:,21:30].hist()

test.iloc[:,31:40].hist()

test.iloc[:,41:50].hist()

test.iloc[:,51:60].hist()

test.iloc[:,61:70].hist()

test.iloc[:,71:80].hist()

test.iloc[:,81:90].hist()

test.iloc[:,101:110].hist()

test.iloc[:,111:120].hist()

test.iloc[:,121:130].hist()

test.iloc[:,131:140].hist()

test.iloc[:,141:150].hist()

test.iloc[:,151:160].hist()

test.iloc[:,161:170].hist()

test.iloc[:,171:180].hist()

test.iloc[:,181:190].hist()

test.iloc[:,191:199].hist()

pyplot.show()
# some baseline features

train['max'] = train.max(axis=1)

train['min'] = train.min(axis=1)

train['mean'] = train.mean(axis=1)

train['sd'] = train.std(axis=1)

test['max'] = test.max(axis=1)

test['min'] = test.min(axis=1)

test['mean'] = test.mean(axis=1)

test['sd'] = test.std(axis=1)
train.head()
test.head()
@jit

def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.335,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': -1

}
#kfold = 15

#folds = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=44000)

num_folds = 11

features = [c for c in train.columns if c not in ['ID_code', 'target']]



folds = KFold(n_splits=num_folds, random_state=2319)

oof = np.zeros(len(train))

getVal = np.zeros(len(train))

predictions = np.zeros(len(train.target))

feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train.target.values)):

    

    X_train, y_train = train.iloc[trn_idx][features], train.target.iloc[trn_idx]

    X_valid, y_valid = train.iloc[val_idx][features], train.target.iloc[val_idx]

    

    X_tr, y_tr = augment(X_train.values, y_train.values)

    X_tr = pd.DataFrame(X_tr)

    

    print("Fold idx:{}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_tr, label=y_tr)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    

    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    getVal[val_idx]+= clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
print("\n >> CV score: {:<8.5f}".format(roc_auc_score(train.target, oof)))
submission = pd.DataFrame({"ID_code": test.ID_code.values})

submission["target"] = predictions

submission.to_csv("submission.csv", index=False)
submission.head()