import numpy as np

import matplotlib.pyplot as plt

np.random.seed(42)



m = 100

X = 4*np.random.rand(m, 1) - 2.5

y = X**3 + 3*X**2 + 2*X + np.random.randn(m, 1)

non_random_y = X**3 + 3*X**2 + 2*X
plt.plot(X, y, 'ro')

plt.plot(X, non_random_y, 'go')

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_converter = PolynomialFeatures(degree=3, include_bias=False)



x_poly_features = poly_converter.fit_transform(X)
X[0]
x_poly_features[0]
X[0]**2
X[0]**3
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_poly_features, y)

lin_reg.coef_
y_pred = lin_reg.predict(x_poly_features)
plt.plot(X, y, 'ro')

plt.plot(X, y_pred, 'bo')

plt.show()
plt.plot(X, non_random_y, 'go')

plt.plot(X, y_pred, 'bo')

plt.show()
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true=non_random_y, y_pred=y_pred)