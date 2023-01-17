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
import lightgbm as lgb

from sklearn.model_selection import cross_validate,KFold,cross_val_score,train_test_split

from sklearn.metrics import make_scorer,roc_auc_score

import xgboost as xgb

from catboost import CatBoostClassifier,Pool,CatBoost

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
def met_auc(y_test,y_pred):

    return roc_auc_score(y_test,y_pred)
stratifiedkfold = KFold(n_splits = 3)
train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv').drop('id',axis=1)
test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv').drop('id',axis=1)
Y = train['G3']

X = train.drop('G3',axis=1)
X_test_d = test.values
import category_encoders as ce



train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv').drop('id',axis=1)

test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv').drop('id',axis=1)



# Eoncodeしたい列をリストで指定。もちろん複数指定可能。

list_cols = ['school','class','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian']



# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')



# pd.DataFrameをそのまま突っ込む

Y = train['G3'].values

train = ce_ohe.fit_transform(train.drop('G3',axis=1))

test = ce_ohe.transform(test)
X = train.values



from sklearn.metrics import mean_squared_error

import math

def rmse_score(y_true, y_pred):

    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""

    mse = mean_squared_error(y_true, y_pred)

    rmse = math.sqrt(mse)

    return rmse



score_func = {'rmse':make_scorer(rmse_score)}



import xgboost as xgb

model = xgb.XGBRegressor()

scores = cross_validate(model, X, Y, cv = stratifiedkfold, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())



import lightgbm as lgb

model = lgb.LGBMRegressor()

scores = cross_validate(model, X, Y, cv = stratifiedkfold, scoring=score_func)

print('rmse:', scores['test_rmse'])

print('rmse:', scores['test_rmse'].mean())
model.fit(X,Y)

p = model.predict(test.values)
sample = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv',index_col = 0)

sample['G3'] = p
sample.to_csv('predict_lightgbm.csv',header = True)
train = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv').drop('id',axis=1)

test = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv').drop('id',axis=1)



X=train.drop('G3',axis=1).values

Y=train['G3'].values

import catboost

model = catboost.CatBoostRegressor(iterations=5000,

                                  use_best_model=True,

                                  eval_metric = 'RMSE')

cat = cat = [0,1,2,4,5,6,9,10,11,12]

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size =0.1, random_state=0)

model.fit(X_train,y_train,

          cat_features=cat,

          eval_set=(X_test,y_test),

          plot=True)
p=model.predict(test.values)
sample = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv',index_col = 0)

sample['G3'] = p
sample.to_csv('predict_catboost.csv',header = True)