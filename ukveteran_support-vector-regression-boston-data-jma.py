import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

import pandas as pd

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

boston_data = load_boston()

df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

df.head()
y = boston_data.target

X = df[['LSTAT']].values
svr = SVR(gamma='auto')

svr.fit(X, y)
sort_idx = X.flatten().argsort()
plt.figure(figsize=(10,8))

plt.scatter(X[sort_idx], y[sort_idx])

plt.plot(X[sort_idx], svr.predict(X[sort_idx]), color='k')



plt.xlabel('LSTAT')

plt.ylabel('MEDV');