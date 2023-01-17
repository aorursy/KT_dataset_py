### Thanks for Mr.Alexandru Papiu and his grate kernel.
### https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
y.isnull().values.any()
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as kNN
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
r1 = kNN()
r2 = Ridge()
r2_1 = Ridge(alpha=0.005)
r2_2 = Ridge(alpha=5)

r3 = RandomForestRegressor(n_estimators=50,n_jobs=-1)
r4 = Lasso(alpha=0.0005)
r4_1 = Lasso()
r4_2 = Lasso(alpha=5)
rx = xgb.XGBRegressor(n_jobs=-1)
r5 = SVR()
r6 = LinearRegression()
regression_stacker = StackingCVRegressor(regressors = [r1,r2,r3,r4,r5,r2_1,r2_2,r4_1,r4_2,rx],meta_regressor = r6, cv=3)

# grid = GridSearchCV(
#     estimator=regression_stacker, 
#     param_grid={
#         'lasso__alpha': [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
#         'ridge__alpha': [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
#     }, 
#     cv=2,
#     refit=True,
#     verbose=100,
#     scoring="neg_mean_squared_error",
#     n_jobs=-1
# )
# grid.fit(X_train.values, y.values)
# print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

regression_stacker.fit(X_train.values,y)
y_pred = regression_stacker.predict(X_test.values)
y_pred_correct = np.expm1(y_pred)
y_pred_correct
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = y_pred_correct
submission.to_csv("submission.csv",index = False)
