# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots

from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df_iris.head()
samples = df_iris[['sepal_length','sepal_width']]

target = df_iris['species']
samples.head()

target.head()
le = preprocessing.LabelEncoder()

le.fit(target.unique())

list(le.classes_)
target_coded = le.transform(target)

target_coded
X_train, X_test, y_train, y_test = train_test_split(samples, target_coded, test_size=0.30)
model = LinearSVC(max_iter=10000)

model.fit(X_train,y_train)

labels = model.predict(X_test)
plt.scatter(X_train['sepal_length'], X_train['sepal_width'], c = y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

xx_min, xx_max = X_train['sepal_length'].min() - 1, X_train['sepal_length'].max() + 1

yy_min, yy_max = X_train['sepal_width'].min() - 1, X_train['sepal_width'].max() + 1

xx, yy = np.meshgrid(np.arange(xx_min, xx_max, 0.02),np.arange(yy_min, yy_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha = 0.1)
plt.scatter(X_test['sepal_length'], X_test['sepal_width'], c = y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

xx_min, xx_max = X_test['sepal_length'].min() - 1, X_test['sepal_length'].max() + 1

yy_min, yy_max = X_test['sepal_width'].min() - 1, X_test['sepal_width'].max() + 1

xx, yy = np.meshgrid(np.arange(xx_min, xx_max, 0.02),np.arange(yy_min, yy_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha = 0.1)
df_spotify = pd.read_csv("../input/spotifyclassification/data.csv")
df_spotify.head()
df_spotify = df_spotify.drop(columns = ['Unnamed: 0','song_title','artist', 'target', 'mode', 'key', 'time_signature'])
df_spotify.dtypes
df_spotify_nomalized = (df_spotify - df_spotify.min())/(df_spotify.max() - df_spotify.min())
df_spotify_nomalized.head()
df_spotify_nomalized.corr()

#valence, danceability

#loudness, acousticness

#energy, acousticness
model2 = KMeans(n_clusters = 5)

model2.fit(df_spotify_nomalized)
labels = model2.labels_

plt.scatter(df_spotify_nomalized['valence'], df_spotify_nomalized['danceability'],c=labels.astype(np.float), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.scatter(df_spotify_nomalized['loudness'], df_spotify_nomalized['acousticness'],c=labels.astype(np.float), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.scatter(df_spotify_nomalized['energy'], df_spotify_nomalized['acousticness'],c=labels.astype(np.float), cmap=plt.cm.coolwarm, s=20, edgecolors='k')