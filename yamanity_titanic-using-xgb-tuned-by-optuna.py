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
# import the other libraries

import optuna # for hyper parameter tuning

from xgboost import XGBClassifier as cls



from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split

from sklearn import datasets

from sklearn.metrics import make_scorer, accuracy_score



from functools import partial
# load data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



# no use columns: 'Name', 'Ticket'

del df_train['Name'], df_train['Ticket'], df_train['Cabin']

del df_test['Name'], df_test['Ticket'], df_test['Cabin']
# preprocess data

df_train = pd.get_dummies(df_train)

df_test  = pd.get_dummies(df_test)
# top 5 of df_train

df_train.head(5)
# top 5 of df_test

df_test.head(5)
df_train_y = df_train['Survived']

del df_train['Survived']

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_train, df_train_y, test_size=0.2, random_state=0)
# objective function

def objective(X, y, trial):

    params = {

        'booster':trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear']),

        'learning_rate':trial.suggest_uniform("learning_rate", 0.0, 1.0),

        'max_depth':trial.suggest_int("max_depth", 2, 30),

        'subsample':trial.suggest_uniform("subsample", 0.0, 1.0),

        'colsample_bytree':trial.suggest_uniform("colsample_bytree", 0.0, 1.0),

    }



    model = cls(**params)

    score = cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score))



    return 1 - score.mean()
f = partial(objective, df_X_train, df_y_train)



study = optuna.create_study()

study.optimize(f, n_trials=100)
# evaluate the model

model = cls(**study.best_params)

model.fit(df_X_train, df_y_train)

y_true = df_y_test

y_pred = model.predict(df_X_test)



print('accuracy[%f]' % accuracy_score(y_true, y_pred))
# make submission file

model = cls(**study.best_params)

model.fit(df_train, df_train_y)

df_test['Survived'] = model.predict(df_test)

df_submit = df_test[['PassengerId', 'Survived']]



df_submit.to_csv('submission.csv', index=False)