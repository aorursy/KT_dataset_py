import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)});



from scipy.stats import normaltest

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import os

print(os.listdir("../input"))
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head(20).T
df.info()
df.describe().T
df['quality'].hist(bins=15);
normaltest(df['quality'])
df['quality_log'] = np.log(df['quality'])

df_target_log = df['quality_log']

print(df['quality'])

print(df_target_log)
scaler = StandardScaler()

df_scaled = df.drop('quality', axis = 1)

df_scaled = df.drop('quality_log', axis = 1)

df_scaled_fin = scaler.fit_transform(df_scaled)

df_scaled_fin
scaler = StandardScaler()



y = df['quality']

X = df.drop('quality', axis=1)

X_new = scaler.fit_transform(X)

print(X_new[:6, :6])
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_new,

                                                      y, 

                                                      test_size=0.2, 

                                                      random_state=42)
knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)
knn.score(X_valid, y_valid)
mean_squared_error(y_valid, y_pred)
kf = KFold(n_splits=5, shuffle=True, random_state=412)

knn = KNeighborsRegressor(n_neighbors=100)

scores = cross_val_score(knn, df_scaled_fin, df_target_log, cv=kf, scoring='neg_mean_squared_error')

scores.mean()
knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, knn_params, scoring='neg_mean_squared_error', cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
results_df = pd.DataFrame(knn_grid.cv_results_)
grid_results = pd.DataFrame(knn_grid.cv_results_)

plt.plot(grid_results['param_n_neighbors'], grid_results['mean_test_score'])

plt.xlabel('n_neighbors')

plt.ylabel('score')

plt.show()
knn2 = KNeighborsClassifier(n_neighbors=1, weights='distance')

knn2_params = {'p': np.linspace(1, 10, num=200, endpoint=True)}

knn2_grid = GridSearchCV(knn2, 

                        knn2_params, 

                        scoring='accuracy',

                        cv=kf)

knn2_grid.fit(X_train, y_train)
knn2_grid.best_params_
knn2_grid.best_score_
grid_results2 = pd.DataFrame(knn2_grid.cv_results_)

plt.plot(grid_results2['param_p'], grid_results2['mean_test_score'])

plt.xlabel('n_neighbors')

plt.ylabel('score')

plt.show()
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()

nc.fit(X_train, y_train)

y3_pred = nc.predict(X_valid)

nc.score(X_valid, y_valid)
accuracy_score(y_valid, y3_pred)