from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/Lotto18-4-18.csv')




dat.head()



dat.describe()




plt.matshow(dat.corr())

plt.colorbar()

plt.show()



plt.scatter(dat['#1'], dat['#2'])




sns.regplot(x=dat['#1'], y=dat['#3'])







sns.lineplot(x='#1', y='#2', data=dat)



plt.style.use('fast')

sns.jointplot(x='#1', y='ext', data=dat)

plt.show()




sns.lineplot(x='#1', y='ext', data=dat)


