import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# generate data

X = np.arange(0,1000).reshape(1000,1)

y = 3*X - 1
print(f'X[max: {X.max()}, min: {X.min()}]')

print(f'y[max: {y.max()}, min: {y.min()}]')

pd.DataFrame({'X': X.flatten(), 'y': y.flatten()}).head(10)
plt.scatter(X,y)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(X,y) # traning
reg.coef_ # coefficient of X
reg.intercept_ # bias
reg.predict([[5252]]) # predicting unseen data
reg.predict([[-2759]]) # predicting unseen data
reg.score