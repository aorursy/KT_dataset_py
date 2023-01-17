"""
Created on Tue Nov 27 20:02:20 2018
Pubg DataSet
@author: Vignesh
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pubg_train=pd.read_csv('../input/train_V2.csv')
pubg_test=pd.read_csv('../input/test_V2.csv')
""" There was just one null value so i just replaced it with a 0"""
pubg_train=pubg_train.fillna(0)
pubg_test=pubg_test.fillna(0) 

"""To Convert the categorical column matchType into factors in order to fit into the model."""
Le=LabelEncoder()
pubg_train['matchType']=Le.fit_transform(pubg_train['matchType'])
pubg_test['matchType']=Le.fit_transform(pubg_test['matchType'])
"""Preparing the data for fitting into the ML model"""

X_train=pubg_train.drop(['winPlacePerc','Id','matchId','groupId'],axis=1)
Y_train=np.array(pubg_train['winPlacePerc'])
Y_train=Y_train.reshape(4446966,1)

"""Lets now fit a Linear Regression model"""
model=LinearRegression()
model=model.fit(X_train,Y_train)
pred=model.predict(X_train)
"""Lets check the accuracy of the model by calculating the R_Square and RMSE"""
rmse=np.sqrt(mean_squared_error(Y_train,pred))
r_sq=model.score(X_train,Y_train)
print("The calculated R_Square and RMSE Values:")
print(r_sq)
print(rmse)
print("The R_square has returned a convincing result of 0.8 which is not a bad value.")
print("The RMSE Score is almost closer to the min value of the column.")
"""Lets now go ahed and predict the winPerc for the Test Case"""
pg_tst=pubg_test.drop(['Id','matchId','groupId'],axis=1)
test_predict=model.predict(pg_tst)
print(test_predict)
print("The output is concatenated just because there are lots of values, around 1lacs<")
"""Lets now try some Hyper Paramater Tuning just to see if it makes any difference in the r_square"""
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters, cv=4)
grid.fit(X_train, Y_train)
print("The tuned R_square value:")
print(grid.best_score_)
print("The tuning process results did not make much difference. Lets just stick on to the same model")