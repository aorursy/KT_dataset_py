# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.datasets

from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error

from sklearn.svm import LinearSVR, NuSVR, SVR

import tensorflow as tf

import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = sklearn.datasets.load_diabetes()

y = data.target

X = data.data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
features = data.feature_names

df = pd.DataFrame(columns=features, data=X)

corr_matrix = df.corr()

sns.set()

f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(corr_matrix, linewidths=.5, annot=True, ax=ax, cmap='coolwarm')

plt.show()
lsvr=LinearSVR()

params = {'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 'tol':[0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5],

         'C':[1, 10, 20, 30, 50, 80, 100]}

search = GridSearchCV(estimator=lsvr, param_grid=params, cv=5)
score = search.fit(X, y)

best_params = score.best_params_
model = LinearSVR(loss=best_params['loss'], C=best_params['C'], tol=best_params['tol'])

model.fit(X_train, y_train)

preds=model.predict(X_test)

linear_mae=mean_absolute_error(preds, y_test)
nn = keras.models.Sequential()



nn.add(keras.layers.Dense(256, activation='relu'))

nn.add(keras.layers.Dense(256, activation='tanh'))

nn.add(keras.layers.Dense(32, activation='relu'))

nn.add(keras.layers.Dense(1, activation='relu'))

nn.compile(optimizer='adam',

          loss='mse')

nn.fit(X_train, y_train, epochs=200, verbose=False)

nn_preds=nn.predict(X_test)

nn_mae = mean_absolute_error(nn_preds,y_test)

print(nn_mae)
nsvr=NuSVR(gamma='auto')

params = {'nu': [0.01, 0.2, 0.4, 0.8, 1], 'tol':[0.0001, 0.01, 0.1],

         'C':[10, 20, 50, 80, 100, 500, 1000], 'kernel': ['rbf', 'poly', 'linear'], 'degree':[2, 3]}

search = GridSearchCV(estimator=nsvr, param_grid=params, cv=5)

score = search.fit(X, y)

best_params = score.best_params_

print(best_params)
nu_model = NuSVR(nu=best_params['nu'], tol=best_params['tol'], C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'])

nu_model.fit(X_train, y_train)

nu_preds = nu_model.predict(X_test)

nu_mae = mean_absolute_error(nu_preds, y_test)
print('Neural Net MAE: {}, LinearSVR MAE: {}, NuSVR MAE: {}'.format(nn_mae, linear_mae, nu_mae))