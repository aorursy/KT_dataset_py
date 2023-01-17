from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



fifa = pd.read_csv('../input/fifa.csv')

iris = pd.read_csv('../input/iris.csv')

spotify = pd.read_csv('../input/spotify.csv')
fifa.head()
fifa.describe()
fifa= pd.read_csv('../input/fifa.csv')

plt.matshow(fifa.corr())

plt.colorbar()

plt.show()
plt.scatter(fifa['BRA'], fifa['ARG'])
plt.scatter(fifa['BRA'], fifa['GER'])
plt.scatter(fifa['BRA'], fifa['ITA'])
iris = pd.read_csv('../input/iris.csv')
iris.describe()
iris.head()
iris= pd.read_csv('../input/iris.csv')

plt.matshow(iris.corr())

plt.colorbar()

plt.show()
plt.scatter(iris['Petal Width (cm)'], iris['Petal Length (cm)'])
plt.scatter(iris['Sepal Width (cm)'], iris['Sepal Length (cm)'])
sns.regplot(x=iris['Sepal Width (cm)'], y=iris['Sepal Length (cm)'])
sns.regplot(iris['Petal Width (cm)'], iris['Petal Length (cm)'])