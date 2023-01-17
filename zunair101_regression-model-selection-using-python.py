import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/power-plant/Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
dataset.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()

regressor.fit(X_poly, y_train)
y_pred = regressor.predict(poly_reg.transform(X_test))

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)