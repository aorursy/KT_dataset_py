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
import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/allstate-claims-severity/train.csv")

test = pd.read_csv("/kaggle/input/allstate-claims-severity/test.csv")

train.shape, test.shape
train.head()
train_cat = train.iloc[:, 1:117]

train_cont = train.iloc[:, 117:-1]
plt.figure(figsize=(16, 150))

for i, col in enumerate(train_cat.columns):

    plt.subplot(30, 4, i+1)

    sns.countplot(train_cat[col], order=train_cat[col].value_counts().sort_index().index)

plt.tight_layout()
plt.figure(figsize=(16, 12))

for i, col in enumerate(train_cont.columns):

    plt.subplot(4, 4, i+1)

    sns.distplot(train_cont[col])

plt.tight_layout()
sns.distplot(np.log1p(train['loss']))
corr = train.drop(columns='id').corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, fmt='.2f', linewidths=0.5)
high_corr= []

threshold = 0.8

for i in range(len(corr)):

    for j in range(i+1, len(corr)):

        if corr.iloc[i,j] >= threshold or (corr.iloc[i, j]<=-threshold and corr.iloc[i, j] < 0):

            high_corr.append([corr.iloc[i,j], i, j])
for v, i, j in high_corr:

    sns.pairplot(train_cont, x_vars=train_cont.columns[i], y_vars=train_cont.columns[j], size= 6)
# In order to make sure train & test sets would have same amount of cols(except loss) after modification



dataset = pd.concat([train, test])

dataset = dataset.drop(columns = ['cont1', 'cont6', 'cont11'])
dataset = pd.get_dummies(dataset)

df_train = dataset[:len(train)]

df_test = dataset[len(train):]

df_test = df_test.drop(columns='loss')
y = np.log1p(df_train['loss'])

df_train = df_train.drop(columns='loss')
from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
x_train, x_test, y_train, y_test = train_test_split(df_train, y, test_size=0.3, random_state=0)
# xgb = XGBRegressor(learning_rate=0.3, n_estimators=500)

# xgb.fit(x_train, y_train)

# mean_absolute_error(np.expm1(y_test), np.expm1(xgb.predict(x_test)))
# xgb=XGBRegressor(seed=18, objective='reg:linear', n_jobs=-1, verbosity=0,

#                        colsample_bylevel=0.764115402027029, colsample_bynode=0.29243734009596956, 

#                        colsample_bytree= 0.7095719673041723, gamma= 4.127534050725986, learning_rate= 0.02387231810322894, 

#                        max_depth=14, min_child_weight=135, n_estimators=828,reg_alpha=0.3170105723222332, 

#                        reg_lambda= 0.3660379465131937, subsample=0.611471430211575)

# xgb.fit(x_train, y_train)

# mean_absolute_error(np.expm1(y_test), np.expm1(xgb.predict(x_test)))
lgb = LGBMRegressor(objective='regression_l1', random_state=18, subsample_freq=1,

                        colsample_bytree=0.3261853512759363, min_child_samples=221, n_estimators=2151, num_leaves= 45, 

                        reg_alpha=0.9113713668943361, reg_lambda=0.8220990333713991, subsample=0.49969995651550947, 

                        max_bin=202, learning_rate=0.02959820893211799)
lgb.fit(x_train, y_train)

mean_absolute_error(np.expm1(y_test), (np.expm1(lgb.predict(x_test))))

# mean_absolute_error(np.expm1(y_test), (np.expm1(lgb.predict(x_test))+np.expm1(xgb.predict(x_test)))/2)
sub = pd.DataFrame({'id': df_test['id'], 'loss': np.expm1(lgb.predict(df_test))})
sub.to_csv('sub.csv', index=False)
from sklearn.model_selection import learning_curve
from hyperopt import hp, fmin, Trials, tpe, pyll
def f(params):

    lgb = LGBMRegressor(**params)

    lgb.fit(x_train, y_train)

    return mean_absolute_error(np.expm1(y_test), (np.expm1(lgb.predict(x_test))))

#     return -cross_val_score(LGBMRegressor(**params), df_train, y, cv=10).mean()



space = {

        'subsample_freq':hp.choice('subsample_freq', range(1, 5)),

        'colsample_bytree':hp.uniform('colsample_bytree', 0.2, 0.5), 

        'min_child_samples':hp.choice('min_child_samples', range(200, 250, 5)), 

        'n_estimators': hp.choice('n_estimators', range(1000, 3000, 100)), 

        'num_leaves': hp.choice('num_leaves', range(20, 50, 5)), 

        'reg_alpha': hp.uniform('reg_alpha', 0.70, 1), 

        'reg_lambda': hp.uniform('reg_lambda', 0.70, 1), 

        'subsample': hp.uniform('subsample', 0.3, 0.6), 

        'max_bin':hp.choice('max_bin', range(150, 250, 5)), 

        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))

}
trial = Trials()

best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trial)
# only idx of best parameters could be achieved from best, so according to space, the values of best parameters could be found



params = {'colsample_bytree':0.2, 'learning_rate': 0.013636902671116896, 'max_bin': 85, 'min_child_samples': 205, 

          'n_estimators': 2000,'num_leaves': 35,'reg_alpha': 0.9579863172141052,'reg_lambda': 0.8783040346489164,

          'subsample': 0.5899650955658289,'subsample_freq': 2}
lgb = LGBMRegressor(**params)

lgb.fit(df_train, y)

sub = pd.DataFrame({'id': df_test['id'], 'loss': np.expm1(lgb.predict(df_test))})

sub.to_csv('sub.csv', index=False)
train_size, train_score, test_score = learning_curve(LGBMRegressor(**params), df_train, y, n_jobs=-1)
train_mean = train_score.mean(axis=1)

train_std = train_score.std(axis=1)

test_mean = test_score.mean(axis=1)

test_std = test_score.std(axis=1)



plt.figure(figsize=(10, 8))

plt.plot(train_size, train_mean, 'o-', linewidth=3)

plt.fill_between(train_size, train_mean+train_std, train_mean-train_std, alpha=0.1)

plt.plot(train_size, test_mean, 'o-', linewidth=3)

plt.fill_between(train_size, test_mean+test_std, test_mean-test_std, alpha=0.1)

plt.title('Learning Curve', size=20)

plt.xlabel('Training Examples')

plt.ylabel('Score')