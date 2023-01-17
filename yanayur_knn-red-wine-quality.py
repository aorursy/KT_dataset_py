import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
(df.info())
df.head(10).T
df['quality']

#целевая переменная
df['quality'].hist(bins = 10);
from scipy.stats import normaltest

data, p = normaltest(df['quality'])

print("p-value = ", p)
df['quality_new'] = np.log(df['quality'])

df_target_new = df['quality_new']

print(df['quality_new'])

print(df['quality_new'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = df.drop('quality', axis = 1)

df_scaled = df.drop('quality_new', axis = 1)

df_scaled_res = scaler.fit_transform(df_scaled)

df_scaled_res
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_scaled_res, 

                                                      df_target_new, 

                                                      test_size=0.3, 

                                                      random_state=50)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=50)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)
knn.score(X_valid, y_valid)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_valid, y_pred)