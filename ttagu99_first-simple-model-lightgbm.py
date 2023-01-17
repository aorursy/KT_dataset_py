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
print(os.listdir("../input/ds2-ds5-competition-1/"))
train = pd.read_csv("../input/ds2-ds5-competition-1/train.csv")

test = pd.read_csv("../input/ds2-ds5-competition-1/test.csv")

submission = pd.read_csv("../input/ds2-ds5-competition-1/sample_submission.csv")
train.info()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(train.label)

plt.xticks(rotation=45)

plt.show() 
len(np.unique(train.label))
sns.countplot(train.label)

plt.xticks(rotation=45)

plt.show() 
neg_ratio = train[train.label == 0].shape[0] / train.shape[0]

print(neg_ratio)
sns.jointplot(x='time', y='label', data=train)

plt.show()
sns.jointplot(x='s1', y='label', data=train)

plt.show()
sns.jointplot(x='s2', y='label', data=train)

plt.show()
sns.jointplot(x='s3', y='label', data=train)

plt.show()
from sklearn.linear_model import LinearRegression
X = train.copy()

x_cols = ['s'+ str(i) for i in list(range(1,17,1))]

X = X[x_cols]

X.head()
print('train.shape:',train.shape)

print('test.shape:',test.shape)
train.head()
train.describe()
from collections import Counter

ct = Counter(train.time)

print('max count time:',max(ct.values()))

tct = Counter(test.time)

print('max count test time:',max(tct.values()))
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression, PLSSVD

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer, MinMaxScaler

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

import lightgbm as lgb
# sp = 900000

# train_df = train.loc[sp:]

# val_df = train.loc[:sp]
# train_df, val_df = train_test_split(train,test_size=0.3, random_state=0, stratify = train.label)

# print(train_df.shape, val_df.shape)
# train_val_df, ho_df = train_test_split(train,test_size=0.1, random_state=0, stratify = train.label)

# print(train_val_df.shape, ho_df.shape)

# train_df, val_df = train_test_split(train_val_df,test_size=0.1, random_state=0, stratify = train_val_df.label)

# print(train_df.shape, val_df.shape)
atrain = train.copy()

atrain['split'] = (train.time/1000).astype(np.int)%5



train_df = atrain[atrain['split']!=0]

val_df = atrain[atrain['split']==0]

mmScaler = MinMaxScaler()

mmScaler.fit(X)



train_s = mmScaler.transform(train_df[x_cols])

val_s = mmScaler.transform(val_df[x_cols])

#ho_s = mmScaler.transform(ho_df[x_cols])

test_s = mmScaler.transform(test[x_cols])

len(train_df)/sum(train_df.label>0)
cls_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'binary_error',

    'learning_rate': 0.01,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    "max_depth": 3,

    "num_leaves": 3**2,

    "scale_pos_weight":len(train_df)/sum(train_df.label>0),

    "n_estimators": 2000

    }



clsgbm = lgb.LGBMClassifier(**cls_params)                 

clsgbm.fit(train_s, train_df.label>0, verbose=100,

        eval_set=[(val_s, val_df.label>0)],

        eval_metric='binary_error',

        early_stopping_rounds=200)
Counter(val_df.label>0)
cls_val_y = clsgbm.predict(val_s, num_iteration=clsgbm.best_iteration_)

print('classification result:',Counter(cls_val_y))
cls_y = clsgbm.predict(test_s, num_iteration=clsgbm.best_iteration_)

print('classification result:',Counter(cls_y))
# from sklearn.linear_model import LinearRegression

# reg = LinearRegression().fit(train_s[train_df.label!=0], train_df.label[train_df.label!=0])

# val_lm_prd = reg.predict(val_s[val_df.label!=0])
# from sklearn.metrics import mean_absolute_error

# mean_absolute_error(val_df.label[val_df.label!=0], val_lm_prd)
hyper_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': ['l1'],

    'learning_rate': 0.1,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 1,

    "max_depth": 3,

    "num_leaves": 3**2, 

    "max_bin": 512,

    "n_estimators": 2000

}

    

gbm = lgb.LGBMRegressor(**hyper_params)

                      

gbm.fit(train_s[train_df.label!=0], train_df.label[train_df.label!=0], verbose=100,

        eval_set=[(val_s[val_df.label!=0], val_df.label[val_df.label!=0])],

        eval_metric='l1',

        early_stopping_rounds=200)
reg_y = gbm.predict(test_s, num_iteration=gbm.best_iteration_)
new_y = reg_y.copy()
for idx, v in enumerate(cls_y):

    if v==False:

        new_y[idx]=0.0
submission_gbm = submission.copy()

cut_new_y = new_y.copy()

for idx, v in enumerate(cut_new_y):

    if v<0:

        cut_new_y[idx]=0.0

    if v>533.33:

        cut_new_y[idx]=533.33

submission_gbm['label'] = cut_new_y
submission_gbm.shape
submission_gbm.head()
submission_gbm.to_csv('submission_gbm.csv', index=False)
max(submission_gbm.label)
# y = train['label']

# lm_model = LinearRegression()

# lm_model.fit(X, y)



# new_X = test[x_cols]

# new_y = lm_model.predict(new_X)

# cut_new_y = new_y.copy()

# for idx, v in enumerate(cut_new_y):

#     if v<0:

#         cut_new_y[idx]=0.0

#     if v>533.33:

#         cut_new_y[idx]=533.33

# submission_lm = submission.copy()

# submission_lm['label'] = cut_new_y

# submission_lm.to_csv('submission_lm.csv', index=False)