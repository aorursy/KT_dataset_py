# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df.head()
df.info()
df.describe().T
## Let's get rid of that ugly column name.

df.rename(columns={"default.payment.next.month":"Target"}, inplace = True)
print(df["MARRIAGE"].unique())

print(df["SEX"].unique())

print(df["EDUCATION"].unique())

print(df["PAY_0"].unique())
df['MARRIAGE'].replace({0 : 3},inplace = True)

df["EDUCATION"].replace({6 : 5, 0 : 5}, inplace = True)

df["PAY_0"].replace({-1 : 0, -2 : 0}, inplace = True)

df["PAY_2"].replace({-1 : 0, -2 : 0}, inplace = True)

df["PAY_3"].replace({-1 : 0, -2 : 0}, inplace = True)

df["PAY_4"].replace({-1 : 0, -2 : 0}, inplace = True)

df["PAY_5"].replace({-1 : 0, -2 : 0}, inplace = True)

df["PAY_6"].replace({-1 : 0, -2 : 0}, inplace = True)
print(df["PAY_0"].unique())

print(df["PAY_2"].unique())

print(df["PAY_3"].unique())

print(df["PAY_4"].unique())

print(df["PAY_5"].unique())

print(df["PAY_6"].unique())
## Original data set might be needed so let's backup it. 

df2 = df.copy()
## .corr generate correlation matrix in df columns.

corr = df.corr()

## np.zeros_like generates matrix which is same shape with correlation matrix so we can use it like mask for inner triangle matrix. 

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1, annot = True, mask = mask)
cat_col = ["SEX","MARRIAGE","EDUCATION"]

df = pd.get_dummies(df, columns = cat_col)
df.drop(columns=["SEX_2","ID"], inplace = True)
## For showing all columns 

pd.set_option('display.max_columns', 50)

df.head()
## Preparing data for pairplotting

df_vis = df[["LIMIT_BAL","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]]

df_vis2 = df[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","AGE"]]
pp1 = sns.pairplot(df_vis)
pp2 = sns.pairplot(df_vis2)
## Pre-processing for building ML model. 

X = df.loc[:,df.columns != "Target"]

Y = df["Target"].copy()
## Importing necessary libraries 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X["BILL_AMT1"] = scaler.fit_transform(np.array(X["BILL_AMT1"]).reshape(-1,1))

X["BILL_AMT2"] = scaler.fit_transform(np.array(X["BILL_AMT2"]).reshape(-1,1))

X["BILL_AMT3"] = scaler.fit_transform(np.array(X["BILL_AMT3"]).reshape(-1,1))

X["BILL_AMT4"] = scaler.fit_transform(np.array(X["BILL_AMT4"]).reshape(-1,1))

X["BILL_AMT5"] = scaler.fit_transform(np.array(X["BILL_AMT5"]).reshape(-1,1))

X["BILL_AMT6"] = scaler.fit_transform(np.array(X["BILL_AMT6"]).reshape(-1,1))

X["PAY_AMT1"] = scaler.fit_transform(np.array(X["PAY_AMT1"]).reshape(-1,1))

X["PAY_AMT2"] = scaler.fit_transform(np.array(X["PAY_AMT2"]).reshape(-1,1))

X["PAY_AMT3"] = scaler.fit_transform(np.array(X["PAY_AMT3"]).reshape(-1,1))

X["PAY_AMT4"] = scaler.fit_transform(np.array(X["PAY_AMT4"]).reshape(-1,1))

X["PAY_AMT5"] = scaler.fit_transform(np.array(X["PAY_AMT5"]).reshape(-1,1))

X["PAY_AMT6"] = scaler.fit_transform(np.array(X["PAY_AMT6"]).reshape(-1,1))

X["LIMIT_BAL"] = scaler.fit_transform(np.array(X["LIMIT_BAL"]).reshape(-1,1))
## Control of the data 

X.head()
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)

selector.fit(X, Y)

k = list(X.columns[selector.get_support(indices=True)])

k

from sklearn.feature_selection import SelectKBest,f_classif

selector = SelectKBest(f_classif, k=10)

selector.fit(X, Y)

k2 = list(X.columns[selector.get_support(indices=True)])

k2
cols_4_model = set(k+k2)

cols_4_model
## Train and test split data

from sklearn.model_selection import train_test_split

X = X[cols_4_model]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
## List of ML Algorithms, we will use for loop for each algorithms.

models = [LogisticRegression(solver = "liblinear"),

          DecisionTreeClassifier(),

          RandomForestClassifier(n_estimators =10),

          XGBClassifier(),

          GradientBoostingClassifier(),

          LGBMClassifier(),

         ]
for model in models:

    t0 = time.time()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    proba = model.predict_proba(X_test)

    roc_score = roc_auc_score(y_test, proba[:,1])

    cv_score = cross_val_score(model,X_train,y_train,cv=10).mean()

    score = accuracy_score(y_test,y_pred)

    bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

    name = str(model)

    print(name[0:name.find("(")])

    print("Accuracy :", score)

    print("CV Score :", cv_score)

    print("AUC Score : ", roc_score)

    print(bin_clf_rep)

    print(confusion_matrix(y_test,y_pred))

    print("Time Taken :", time.time()-t0, "seconds")

    print("------------------------------------------------------------")
## LGBM_CLF Model

t0 = time.time()

lgbm_model = LGBMClassifier()

lgbm_model.fit(X_train,y_train)

y_pred = lgbm_model.predict(X_test)

proba = lgbm_model.predict_proba(X_test)

roc_score = roc_auc_score(y_test, proba[:,1])

cv_score = cross_val_score(lgbm_model,X_train,y_train,cv=10).mean()

score = accuracy_score(y_test,y_pred)

bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

print(name[0:name.find("(")])

print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", roc_score)

print(bin_clf_rep)

print(confusion_matrix(y_test,y_pred))

print("Time Taken :", time.time()-t0, "seconds")

lgbm_model
## Setting parameters for LGBM Model, we will use this dictionary with GridSearchCV

lgbm_params = {"n_estimators" : [100, 500, 1000],

               "subsample" : [0.6, 0.8, 1.0],

               "learning_rate" : [0.1, 0.01, 0.02],

               "min_child_samples" : [5, 10, 20]}
## n_jobs = -1 allows multicore processing for CPU

from sklearn.model_selection import GridSearchCV

lgbm_cv_model = GridSearchCV(lgbm_model, 

                             lgbm_params, 

                             cv = 5,

                             verbose = 1,

                             n_jobs = -1)
## Code works approximately 5-6 minutes

lgbm_cv_model.fit(X_train, y_train)
## Getting best parameters

lgbm_cv_model.best_params_
## Best_Params with LGBM_CLF

t0 = time.time()

lgbm_model2 = LGBMClassifier(learning_rate = 0.01,

                            min_child_samples = 20,

                            n_estimators = 500,

                            subsample = 0.6)

lgbm_model2.fit(X_train,y_train)

y_pred = lgbm_model2.predict(X_test)

proba = lgbm_model2.predict_proba(X_test)

roc_score = roc_auc_score(y_test, proba[:,1])

cv_score = cross_val_score(lgbm_model2,X_train,y_train,cv=10).mean()

score = accuracy_score(y_test,y_pred)

bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

print(name[0:name.find("(")])

print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", roc_score)

print(bin_clf_rep)

print(confusion_matrix(y_test,y_pred))

print("Time Taken :", time.time()-t0, "seconds")
## LOG_REG_CLF

t0 = time.time()

log_reg_model = LogisticRegression(solver="liblinear")

log_reg_model.fit(X_train,y_train)

y_pred = log_reg_model.predict(X_test)

proba = log_reg_model.predict_proba(X_test)

roc_score = roc_auc_score(y_test, proba[:,1])

cv_score = cross_val_score(log_reg_model,X_train,y_train,cv=10).mean()

score = accuracy_score(y_test,y_pred)

bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

print(name[0:name.find("(")])

print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", roc_score)

print(bin_clf_rep)

print(confusion_matrix(y_test,y_pred))

print("Time Taken :", time.time()-t0, "saniye")
log_reg_params = {"C":[0.1, 0.5, 1.0], 

                  "penalty":["l1","l2"],

                  "solver" : ["liblinear", "lbfgs", "newton-cg"],

                  "max_iter" : [100,200,500]

                  }
from sklearn.model_selection import GridSearchCV

log_reg_cv_model = GridSearchCV(log_reg_model, 

                             log_reg_params, 

                             cv = 5,

                             verbose = 1,

                             n_jobs = -1)
log_reg_cv_model.fit(X_train, y_train)
log_reg_cv_model.best_params_
## Best Params with LOG_REG_CLF

t0 = time.time()

log_reg_model = LogisticRegression(solver="liblinear",

                                  C = 0.5,

                                  max_iter = 100,

                                  penalty = "l1",

                                  )

log_reg_model.fit(X_train,y_train)

y_pred = log_reg_model.predict(X_test)

proba = log_reg_model.predict_proba(X_test)

roc_score = roc_auc_score(y_test, proba[:,1])

cv_score = cross_val_score(log_reg_model,X_train,y_train,cv=10).mean()

score = accuracy_score(y_test,y_pred)

bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

print(name[0:name.find("(")])

print("Accuracy :", score)

print("CV Score :", cv_score)

print("AUC Score : ", roc_score)

print(bin_clf_rep)

print(confusion_matrix(y_test,y_pred))

print("Time Taken :", time.time()-t0, "saniye")