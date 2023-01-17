%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
w = 5

b = 1

sigma = 3
X_10 = np.arange(10)

X_10
Y_10 = np.array([w*i + b + sigma*np.random.randn() for i in X_10])

Y_10
plt.scatter(X_10, Y_10)
X_30 = np.arange(30)

Y_30 = np.array([w*i + b + sigma*np.random.randn() for i in X_30])
plt.scatter(X_30, Y_30)
X_100 = np.arange(100)

Y_100 = np.array([w*i + b + sigma*np.random.randn() for i in X_100])
plt.scatter(X_100, Y_100)
X_200 = np.arange(200)

Y_200 = np.array([w*i + b + sigma*np.random.randn() for i in X_200])
plt.scatter(X_200, Y_200)
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_10.reshape(-1,1), Y_10.reshape(-1,1))
print('w = ', w)

w_10 = linear_regression.coef_

print('w_10 = ', w_10)
linear_regression = LinearRegression().fit(X_30.reshape(-1,1), Y_30.reshape(-1,1))

print('w = ', w)

w_30 = linear_regression.coef_

print('w_30 = ', w_30)
linear_regression = LinearRegression().fit(X_100.reshape(-1,1), Y_100.reshape(-1,1))

print('w = ', w)

w_100 = linear_regression.coef_

print('w_100 = ', w_100)
linear_regression = LinearRegression().fit(X_200.reshape(-1,1), Y_200.reshape(-1,1))

print('w = ', w)

w_200 = linear_regression.coef_

print('w_200 = ', w_200)