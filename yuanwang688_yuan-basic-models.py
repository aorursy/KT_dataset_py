# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import kagglegym

import numpy as np

from matplotlib import pyplot as plt

plt.plot()
env = kagglegym.make()

observations = env.reset()

df = observations.train

df_normed = (df - df.mean()) / (df.max() - df.min())

df_normed['id'] = df['id']

df_normed['timestamp'] = df['timestamp']

df = df_normed
id_len = []

for id in df['id'].unique():

    id_len += [[id, len(df[df['id']==id])]]
id_len = np.array(id_len)
id_len_max = np.argmax(id_len, axis=0)
print(id_len_max)
id_len[1]
df_11 = df[df['id']==11]
df_11
df_11_dropna = df_11.dropna(axis=1)
df_11_dropna['y'].plot()
df_11_dropna_train = df_11_dropna[:500]

df_11_dropna_valid = df_11_dropna[500:]
df_11_dropna_train['y'].plot()
df_11_dropna_valid['y'].plot()
feature_list = df_11_dropna_train.columns.tolist()

feature_list = feature_list[2:-1]

print(feature_list)
y_train = df_11_dropna_train['y'].values

X_train = df_11_dropna_train[feature_list].values
y_valid = df_11_dropna_valid['y'].values

X_valid = df_11_dropna_valid[feature_list]
#linear regression model

from sklearn import linear_model

lm = linear_model.Ridge(alpha=1)

lm.fit(X_train, y_train)

print('Train: {}'.format(lm.score(X_train, y_train)))

print('Valid: {}'.format(lm.score(X_valid, y_valid)))
#visualize linear regression model

y_train_pred = lm.predict(X_train)

plt.plot(range(len(y_train)), y_train, range(len(y_train)), y_train_pred)
y_valid_pred = lm.predict(X_valid)

plt.plot(range(len(y_valid)), y_valid, range(len(y_valid)), y_valid_pred)
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=[100, 50, 10], activation='tanh')

mlp.fit(X_train, y_train)

print('Train: {}'.format(mlp.score(X_train, y_train)))

print('Valid: {}'.format(mlp.score(X_valid, y_valid)))
y_train_pred = mlp.predict(X_train)

plt.plot(range(len(y_train)), y_train, range(len(y_train)), y_train_pred)
y_valid_pred = mlp.predict(X_valid)

plt.plot(range(len(y_valid)), y_valid, range(len(y_valid)), y_valid_pred)
corr=df_11.corr()
corr['y'].plot()
knn = neighbors.KNeighborsRegressor(20, weights='uniform')

knn.fit(X_train, y_train)

print('Train: {}'.format(knn.score(X_train, y_train)))

print('Valid: {}'.format(knn.score(X_valid, y_valid)))
y_train_pred = knn.predict(X_train)

plt.plot(range(len(y_train)), y_train, range(len(y_train)), y_train_pred)
y_valid_pred = knn.predict(X_valid)

plt.plot(range(len(y_valid)), y_valid, range(len(y_valid)), y_valid_pred)
from sklearn import ensemble

ada = ensemble.AdaBoostRegressor()

ada.fit(X_train, y_train)

print('Train: {}'.format(ada.score(X_train, y_train)))

print('Valid: {}'.format(ada.score(X_valid, y_valid)))
y_train_pred = ada.predict(X_train)

plt.plot(range(len(y_train)), y_train, range(len(y_train)), y_train_pred)
y_valid_pred = ada.predict(X_valid)

plt.plot(range(len(y_valid)), y_valid, range(len(y_valid)), y_valid_pred)
rf = ensemble.RandomForestRegressor(n_estimators=100, min_samples_split=25, min_samples_leaf=5)

rf.fit(X_train, y_train)

print('Train: {}'.format(rf.score(X_train, y_train)))

print('Valid: {}'.format(rf.score(X_valid, y_valid)))
y_train_pred = rf.predict(X_train)

plt.plot(range(len(y_train)), y_train, range(len(y_train)), y_train_pred)
y_valid_pred = rf.predict(X_valid)

plt.plot(range(len(y_valid)), y_valid, range(len(y_valid)), y_valid_pred)
import pykalman