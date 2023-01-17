# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Загружаем данные
df = pd.read_csv('../input/adult-income-dataset/adult.csv', sep=',')
df.describe()
df.head(10)
df.dtypes

df['income'].value_counts()
df['income'].value_counts().plot(kind='bar', color='black')
df['income']=df['income'].map({ '>50K': 1, '<=50K': 0})
from scipy.stats import normaltest
normaltest(df['income'])
df['income'].value_counts().plot(kind='bar', color='black')
df.head()
df=df.drop(["fnlwgt"],axis=1)
df=df.drop(["education"],axis=1)
df = df[df["workclass"] != "?"]
df = df[df["occupation"] != "?"]
df = df[df["native-country"] != "?"]
df = pd.get_dummies(df, columns=['workclass', 'occupation', 'native-country','marital-status', 'relationship']) 
df.head()
df1 =['race','gender'] 
for i in df1:
    unique_value, index = np.unique(df[i], return_inverse=True) 
    df[i] = index
df.head(10)
from sklearn.model_selection import train_test_split
X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.2, random_state=28)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
neighbors = KNeighborsClassifier(n_neighbors=5)
neighbors.fit(X_train, y_train)

y_pred = neighbors.predict(X_valid)
print(accuracy_score(y_valid, y_pred))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=28)
neighbors = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(neighbors, X, y, cv=kf, scoring='accuracy')
print('Массив значений метрики:', scores)
print('Средняя метрика на кросс-валидации:', np.mean(scores))
from sklearn.model_selection import GridSearchCV
neighbors_params = {'n_neighbors': np.arange(1, 50)} # словарь параметров (ключ: набор возможных значений)
neighbors_grid = GridSearchCV(neighbors, neighbors_params, cv=kf, scoring='accuracy')  # кросс-валидация по 5 блокам
neighbors_grid.fit(X_train, y_train)
neighbors_grid.best_params_
neighbors_grid.best_score_
pd.DataFrame(neighbors_grid.cv_results_).T
# По оси х --- кол-во ближайших соседей
# По оси y --- точность угадывания
import matplotlib.pyplot as plt

results_df = pd.DataFrame(neighbors_grid.cv_results_)
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])
plt.figure()
p_params = {'p': np.linspace(1,10,200)}
neighbors = KNeighborsClassifier(n_neighbors=18, weights = 'distance', n_jobs = -1)
cv = GridSearchCV(neighbors, p_params, cv = kf, scoring='accuracy', verbose = 100)
cv.fit(X,y)
cv.best_params_
neighbors_grid.best_score_
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid()
nc.fit(X_train, y_train)

nc.score(X_valid, y_valid)
from sklearn.neighbors import RadiusNeighborsClassifier
neigh = RadiusNeighborsClassifier(radius=500.0)
neigh.fit(X_train, y_train)
neigh.score(X_valid,y_valid)
neigh2 = RadiusNeighborsClassifier(radius=1500.0)
neigh2.fit(X_train, y_train)
neigh2.score(X_valid,y_valid)
neigh2 = RadiusNeighborsClassifier(radius=1000.0)
neigh2.fit(X_train, y_train)
neigh2.score(X_valid,y_valid)