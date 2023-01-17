# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head()
from sklearn.model_selection import train_test_split



X, y = df.iloc[:,:-1],df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import xgboost as xgb

from xgboost import XGBClassifier



xg_reg = XGBClassifier(learning_rate = 0.02,

                max_depth = 5, n_estimators = 500)

                           

xg_reg.fit(X_train,y_train)



y_pred = xg_reg.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_pred, y_test)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

      

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



folds = 5

param_comb = 100



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)



random_search = RandomizedSearchCV(xg_reg, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=42 )



random_search.fit(X_train, y_train)
print('\n Best estimator:')

print(random_search.best_estimator_)
random_search.best_estimator_.score(X_test,y_test)