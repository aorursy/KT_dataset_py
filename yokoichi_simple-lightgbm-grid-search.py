import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

import lightgbm as lgb

from sklearn.model_selection import cross_val_score,KFold

train = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv")

test = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_test.csv')

train_num = train.shape[0]

data = pd.concat([train,test])



def dummy_date(df):

    df["year"] = df["Date"].apply(lambda x: x.split("-")[0])

    df["month"] = df["Date"].apply(lambda x: x.split("-")[1])

    df["day"] = df["Date"].apply(lambda x: x.split("-")[2])

    #df.drop("Date",inplace=True,axis=1)

    return df



def LabelEncord_categorical(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        le = LabelEncoder()

        df[params] = le.fit_transform(df[params])

    return df



def dummies(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        dummies =  pd.get_dummies(df[params])

        df = pd.concat([df, dummies],axis=1)

    return df



def pre_processing(df):

    df = dummy_date(df)

    df = LabelEncord_categorical(df)

    df = dummies(df)

    return df



data = pre_processing(data)



train = data[:train_num]

test = data[train_num:]
y_train = train["Price"].values

X_train = train.drop(["Price","Date"],axis=1).values

y_test = test["Price"].values

X_test =test.drop(["Price","Date"],axis=1).values

from sklearn.model_selection import train_test_split

# train_df, val_df = train_test_split(train_df, test_size=.10, random_state=1)
grid_param ={'n_estimators': [100],'max_depth':[-1, 2,3,4,5, ],'num_leaves':[3, 7, 15, 31],'learning_rate':[0.1,0.05,0.01]}



fit_params={'early_stopping_rounds':10, 

            'eval_metric' : 'rmse', 

            'eval_set' : [(X_train, y_train)]

           }
from sklearn.model_selection import GridSearchCV



bst = lgb.LGBMRegressor(

                        num_leaves = 31,

                        learning_rate=0.01,

                        min_child_samples=10,

                        n_estimators=1000,

                        max_depth=-1,

                        )





bst_gs_cv = GridSearchCV(

            bst, # 識別器

            grid_param, # 最適化したいパラメータセット 

            verbose = 1

            )



bst_gs_cv.fit(

            X_train, 

            y_train,

            **fit_params,

            verbose = 0

            )



best_param = bst_gs_cv.best_params_

print('Best parameter: {}'.format(best_param))
bgt = lgb.LGBMRegressor(**bst_gs_cv.best_params_)

bgt.fit(X_train,y_train)
predictions = bgt.predict(X_test)

predictions
bgt.score(X_train,y_train)
submission = pd.DataFrame({ 'DATE': test['Date'],

                            'PRICE': predictions })

submission.head()
submission.to_csv('submission.csv', index=False)