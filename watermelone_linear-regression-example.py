import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error

#Define a y array with the right answers

y = [4, 0, 8, 4, 14, 18, 22, 17, 22, 28, 29, 33, 36, 41, 42]

x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14]]

regr = linear_model.LinearRegression()

regr.fit(x, y)

y_pred = regr.predict(x)

plt.scatter(x, y)

plt.plot(x, y_pred)

plt.show()

#Make a predictions array called y_pred using your model and the .predict(x) function

#Plot the outputs. Use plt.scatter(x, y) for the true values, and use plt.plot(x, y_pred) for the predicted line.

#plt.show() to show the graph

#If you have extra time, use the mean_squared_error(y, y_pred) function to calculate your mean squared error
