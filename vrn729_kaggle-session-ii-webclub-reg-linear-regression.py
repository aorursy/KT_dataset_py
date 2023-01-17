# Import Pandas for data manipulation and analysis

import pandas as pd

# Import NumPy for algebra, matrix and other mathematical operations

import numpy as np

# Import matplotlib for visualizations and graphs

import matplotlib

import matplotlib.pyplot as plt

# Import skew to compute the skewness

from scipy.stats import skew



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
# Combining the train and the test sets to perform Data pre-processing

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data.head()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
# log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



# log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index



# Compute the skewness of the numeric features

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))



# Take the features which have skewness more than 0.75

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



# Apply logarithmic function to them

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# Creating dummy variables for non-numeric features

all_data = pd.get_dummies(all_data)
# Filling NANs and missing values with the mean of the column

all_data = all_data.fillna(all_data.mean())
# Creating separte matrices for training and testing

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]



# Target values for train set

y = train.SalePrice
# Importing models and cross-validation

# Linear regression and its regularization variants

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV



# Random forest regression model

from sklearn.ensemble import RandomForestRegressor



# For Cross-Validation and Hyperparameter tuning

from sklearn.model_selection import cross_val_score, GridSearchCV
# Defining a function for estimating the rmse error on running 5-fold Cross-Validation

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring = "neg_mean_squared_error", cv = 5))

    return (rmse)
# Linear regression without Regularization

model_LR = LinearRegression()
# Error for the training data

display(rmse_cv(model_LR).mean())
# Linear regression using L2 Regularization

model_ridge = Ridge()
# Check for best value of regularization parameter lambda

lambdas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = lmd)).mean() for lmd in lambdas] # alpha is another name for lambda
# one-dimensional array with lambda values as labels

pd.Series(cv_ridge, index = lambdas)
cv_ridge = pd.Series(cv_ridge, index = lambdas)

cv_ridge.plot(title = "Validation")

plt.xlabel("Regularization parameter (lambda)")

plt.ylabel("rmse")
# minimum error

cv_ridge.min()
# Using L1 regularization

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], cv= 5).fit(X_train, y) # alphas is another name for lambdas
arr = rmse_cv(model_lasso)

display(arr)

display(arr.mean())
# Using L1 and L2 regularization

model_elastic = ElasticNetCV(alphas = [1, 0.1, 0.001, 0.0005], cv= 5).fit(X_train, y) # alphas is another name for lambdas
arr = rmse_cv(model_elastic)

display(arr)

display(arr.mean())
# Using Random Forest Regression

model_rfr = RandomForestRegressor()
# Using GridSearchCV to find the best parameters for the model

gsc = GridSearchCV(estimator = model_rfr, param_grid = {'n_estimators': [10, 30, 50, 100],

                                                        'max_depth': range(2, 7)}, scoring = 'neg_mean_squared_error', cv = 3)

grid_result = gsc.fit(X_train, y)

best_params = grid_result.best_params_

display(best_params)
# Defining a model using the best parameters

rfr = RandomForestRegressor(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'])
arr = rmse_cv(rfr)

display(arr)

display(arr.mean())
X_test.head()
ridge = Ridge(alpha = 10)

ridge.fit(X_train, y)

pred = ridge.predict(X_test)

pred = np.exp(pred)
result=pd.DataFrame()

result['Id'] = pd.Series(np.arange(len(pred)))

result['SalePrice'] = pd.Series(pred)

result.head()
result.to_csv('model.csv', index = False)
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)
from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape
X_tr
model = Sequential()

#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))

model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))



model.compile(loss = "mse", optimizer = "adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))