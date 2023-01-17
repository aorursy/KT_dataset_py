import numpy as np

import pandas as pd

import pandas_profiling as pdp

import matplotlib.pyplot as plt

%matplotlib inline

import xgboost as xgb

from sklearn.model_selection import GridSearchCV 

from sklearn.preprocessing import LabelEncoder
shopping_df = pd.read_csv('../input/shoppingdata/shopping-data.csv')

shopping_profile_df = pdp.ProfileReport(shopping_df)
shopping_profile_df
le_gender = LabelEncoder()

shopping_df.Genre = le_gender.fit_transform(shopping_df.Genre)

X = shopping_df.iloc[:,:-1].values

y = shopping_df.iloc[:,-1].values
objective = "reg:linear"

seed = 100

n_estimators = 100

learning_rate = 0.2

gamma = 0.1

subsample = 0.8

colsample_bytree = 0.8

reg_alpha = 1

reg_lambda = 1

silent = False



parameters = {}

parameters['objective'] = objective

parameters['seed'] = seed

parameters['n_estimators'] = n_estimators

parameters['learning_rate'] = learning_rate

parameters['gamma'] = gamma

parameters['colsample_bytree'] = colsample_bytree

parameters['reg_alpha'] = reg_alpha

parameters['reg_lambda'] = reg_lambda

parameters['silent'] = silent



scores = []



cv_params = {'max_depth': [i for i in range(2,20)],

             'min_child_weight': [i for i in range(1,45)]

            }



gbm = GridSearchCV(xgb.XGBRegressor(

                                        objective = objective,

                                        seed = seed,

                                        n_estimators = n_estimators,

                                        learning_rate = learning_rate,

                                        gamma = gamma,

                                        subsample = subsample,

                                        colsample_bytree = colsample_bytree,

                                        reg_alpha = reg_alpha,

                                        reg_lambda = reg_lambda,

                                        silent = silent



                                    ),

                    

                    param_grid = cv_params,

                    iid = False,

                    scoring = "neg_mean_squared_error",

                    cv = 5,

                    verbose = True

)



gbm.fit(X,y)
print("Best score: ",gbm.best_score_)

print("Best params: ",gbm.best_params_)