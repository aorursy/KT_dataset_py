"""

Kaggle Titanic



Load original datasets

"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train_org = pd.read_csv('../input/train.csv')

test_org = pd.read_csv('../input/test.csv')



import pandas_profiling as pdp

profile = pdp.ProfileReport(train_org)

profile.to_file(outputfile="train_profile.html")

profile = pdp.ProfileReport(test_org)

profile.to_file(outputfile="test_profile.html")
"""

Select easily usable columns and convert and fill values if necessary

"""

y_column = "Survived"

y_train = train_org[y_column]

x_columns = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df = pd.concat([train_org[x_columns], test_org[x_columns]])

for cname in df.columns:

  col = df[cname]

  if col.dtype == np.dtype('O'):

    print("Onehot:",cname)

    onehot = pd.get_dummies(col, cname)

    df = pd.concat([df.drop(cname,axis=1), onehot], axis=1)

  else:

    nans = col.isnull()

    if nans.any():

      print("Nan:", cname)

      df[cname+"_NaN"] = nans.astype(int)

df = df.fillna(df.median())



X_train = df[:len(train_org)]

X_test = df[-len(test_org):]

profile = pdp.ProfileReport(X_train)

profile.to_file(outputfile="X_train_profile.html")

profile = pdp.ProfileReport(X_test)

profile.to_file(outputfile="X_test_profile.html")
"""

Find the best model with XGB and GridSearch

"""

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



clf = xgb.XGBClassifier()

clf_cv = GridSearchCV(clf, {'max_depth': [4,5,6], 'n_estimators': [32,64,96,128,160,192,224,256]}, cv=5, n_jobs=-1, scoring="roc_auc", verbose=1)

clf_cv.fit(X_train, y_train, eval_metric=['auc'])

print(clf_cv.best_params_, clf_cv.best_score_)
"""

Evaluate the model

"""

y_pred = clf_cv.predict(X_train)

acc = metrics.accuracy_score(y_train, y_pred)

proba = clf_cv.predict_proba(X_train)

predprob = proba[:,1]

auc = metrics.roc_auc_score(y_train, predprob)

print("Train score Accuracy: %f, AUC: %f" % (acc,auc))
"""

Output a final result set

"""

y_pred = clf_cv.predict(X_test)

csv = pd.concat([X_test["PassengerId"], pd.DataFrame(y_pred, columns=[y_column])], axis=1)

csv.to_csv("result.csv", index=False)