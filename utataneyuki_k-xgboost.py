import pandas as pd

import math

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

%matplotlib inline

font = {"family":"HGMaruGothicMPRO"}

matplotlib.rc("font",**font)

import seaborn as sns

from pandas import Series,DataFrame

import numpy as np

import re
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

dfs =  pd.read_csv('../input/train_small.csv')
df.info()
def replace_outlier(series, bias=1.5):

    #四分位数

    q1 = series.quantile(.25)

    q3 = series.quantile(.75)

    iqr = q3 - q1



    #外れ値の基準点

    outlier_min = q1 - (iqr) * bias

    outlier_max = q3 + (iqr) * bias



    print("outlier_min :" + str(outlier_min) + ", outlier_max :" + str(outlier_max))



    #外れ値をクリップする

    series = series.clip(outlier_min, outlier_max)

    return series
df2 = df[0:300000]

#df2 = df

#df2=df.sample(n=90000)



y = df2[["loan_condition"]]
df2 = df2.drop(["ID","loan_condition","emp_title", "zip_code","emp_length","home_ownership","purpose","addr_state",

                            "initial_list_status","application_type"], axis=1)
test = test.drop(["ID","emp_title","zip_code","emp_length","home_ownership","purpose","addr_state",

                            "initial_list_status","application_type"], axis=1)
df2["issue_d"] = df2["issue_d"][4:]

df2["title"] = df2["title"].str[0:3]

df2["earliest_cr_line"] = df2["earliest_cr_line"].str[4:]

test["issue_d"] = test["issue_d"][4:]

test["title"] = test["title"].str[0:3]

test["earliest_cr_line"] = test["earliest_cr_line"].str[4:]

df2["annual_inc"] = replace_outlier(df2["annual_inc"])

df2["revol_bal"] = replace_outlier(df2["revol_bal"])

dfs["tot_cur_bal"] = replace_outlier(dfs["tot_cur_bal"])

dfs["tot_coll_amt"] = replace_outlier(dfs["tot_coll_amt"])

test["annual_inc"] = replace_outlier(test["annual_inc"])

test["revol_bal"] = replace_outlier(test["revol_bal"])

test["tot_cur_bal"] = replace_outlier(test["tot_cur_bal"])
#ありあえないほどダミー化の処理が長いので別の手法を考える

#グレード、サブグレードをある程度まとめるor数値で扱う

#emp_lenghtを数字だけにしてデータ型も変更

#titleを形が近いものでまとめる "Thank you" など出現回数が少ないものはその他に

#回帰で欠損値を埋める

#必要に応じて標準化

#issue_dには月、年が記載しているので分けて使うかdatetimeに変換？
X2 = pd.get_dummies(columns=["grade","sub_grade","issue_d","title","earliest_cr_line"],data=df2)

test2 = pd.get_dummies(columns=["grade","sub_grade","issue_d","title","earliest_cr_line"],data=test)
#X2['emp_title'].fillna(X2['emp_title'].mode())

#X2['emp_length'].fillna(X2['emp_length'].mode())

X2['dti'].fillna(X2['dti'].median(),inplace = True)

X2['revol_util'].fillna(X2['revol_util'].median(),inplace = True)

X2['mths_since_last_delinq'].fillna(0,inplace = True)

X2['mths_since_last_record'].fillna(0,inplace = True)

X2['mths_since_last_major_derog'].fillna(0,inplace = True)

X2['tot_coll_amt'].fillna(0,inplace = True)

X2['tot_cur_bal'].fillna(X2['tot_cur_bal'].median(),inplace = True)



#X3 = X2.drop("emp_title",axis=1)

#X3 = X3.drop("emp_length",axis=1)
X2['annual_inc'].fillna(X2['annual_inc'].median(),inplace = True)

X2['delinq_2yrs'].fillna(X2['delinq_2yrs'].median(),inplace = True)

X2['inq_last_6mths'].fillna(0,inplace = True)

X2['open_acc'].fillna(0,inplace = True)

X2['pub_rec'].fillna(0,inplace = True)

X2['total_acc'].fillna(0,inplace = True)

X2['collections_12_mths_ex_med'].fillna(X2['collections_12_mths_ex_med'].median(),inplace = True)

X2['acc_now_delinq'].fillna(X2['acc_now_delinq'].median(),inplace = True)
test2['dti']=test2['dti'].fillna(test2['dti'].median())

test2['mths_since_last_delinq']=test2['mths_since_last_delinq'].fillna(0)

test2['mths_since_last_record']=test2['mths_since_last_record'].fillna(0)

test2['mths_since_last_major_derog']=test2['mths_since_last_major_derog'].fillna(0)

test2['tot_coll_amt']=test2['tot_coll_amt'].fillna(0)

test2['tot_cur_bal']=test2['tot_cur_bal'].fillna(test2['tot_cur_bal'].median())

test2['inq_last_6mths']=test2['inq_last_6mths'].fillna(test2['inq_last_6mths'].median())

test2['revol_util']=test2['revol_util'].fillna(test2['revol_util'].median())

#test3 = test2.drop("emp_title",axis=1)

#test3 = test3.drop("emp_length",axis=1)
m_col=set(X2.columns)-set(test2.columns)

for c in m_col:

    test2[c]=0



m_col2=set(test2.columns)-set(X2.columns)

for c in m_col2:

    X2[c]=0
#from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
model = xgb.XGBClassifier(max_depth = 5,cv=4)
X2 = X2.sort_index(axis=1)
X2.info()
test2 = test2.sort_index(axis=1)
test2.info()
X_train,X_test,y_train,y_test = train_test_split(X2,y)

model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score

roc_auc_score(y_test,pred)
pred_out = model.predict_proba(test2)[:,1]
pred_out
submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = pred_out

submission.to_csv('submission.csv')