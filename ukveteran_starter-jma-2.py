from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



hepatitis = pd.read_csv('../input/hepatitis.csv')

measles = pd.read_csv('../input/measles.csv')

mumps = pd.read_csv('../input/mumps.csv')

pertussis = pd.read_csv('../input/pertussis.csv')

polio = pd.read_csv('../input/polio.csv')

rubella = pd.read_csv('../input/rubella.csv')

smallpox = pd.read_csv('../input/smallpox.csv')
hepatitis.head()
hepatitis.info()
hepatitis.describe()
sns.boxplot(hepatitis['cases'])
sns.boxplot(hepatitis['incidence_per_capita'])
plt.scatter(hepatitis['cases'], hepatitis['incidence_per_capita'])
mumps = pd.read_csv('../input/mumps.csv')

mumps.describe()
plt.scatter(mumps['cases'], mumps['incidence_per_capita'])
smallpox = pd.read_csv('../input/smallpox.csv')

smallpox.describe()

plt.scatter(smallpox['cases'], smallpox['incidence_per_capita'])
smallpox = pd.read_csv('../input/smallpox.csv')

plt.matshow(smallpox.corr())

plt.colorbar()

plt.show()
hepatitis = pd.read_csv('../input/hepatitis.csv')

plt.matshow(hepatitis.corr())

plt.colorbar()

plt.show()