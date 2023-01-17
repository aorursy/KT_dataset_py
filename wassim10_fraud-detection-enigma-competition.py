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
data=pd.read_csv('../input/train.csv')
data.head()
pd.value_counts(data.target, normalize = True)
from imblearn.under_sampling import RandomUnderSampler

X = data.drop(['target','id'],axis = 1)

y = data.target

rus = RandomUnderSampler(sampling_strategy=0.8)

X_res, y_res = rus.fit_resample(X, y)

print(X_res.shape, y_res.shape)

print(pd.value_counts(y_res))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=23)
from sklearn.metrics import roc_auc_score

import xgboost as xgb

params={

    'max_depth': [2,4], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env

    'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],

    'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],

    'n_estimators': [1000,2000], #[1000,2000,3000]

    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]

}



xgb_clf = xgb.XGBClassifier(missing=9999999999)

rs = GridSearchCV(xgb_clf,

                  params,

                  cv=5,

                  scoring="roc_auc",

                  n_jobs=-1,

                  verbose=2)

rs.fit(X_train, y_train)

best_est = rs.best_estimator_

print(rs.best_score_)



validation = best_est.predict_proba(X_train)

print(validation)

print("Roc AUC: ", roc_auc_score(y_train, validation[:,1], average='macro'))
y_pred=best_est.predict_proba(X_test)

print("Roc AUC: ", roc_auc_score(y_test, y_pred[:,1], average='macro'))
y_p=best_est.predict(X_test)
from sklearn.metrics import classification_report

print(accuracy_score(y_test,y_p))

print(classification_report(y_test, y_p))