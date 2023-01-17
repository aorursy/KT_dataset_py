from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd
hs = pd.read_csv('../input/kc_house_data.csv')
hs.head()
hs.describe()
plt.matshow(hs.corr())

plt.colorbar()

plt.show()
sns.lineplot(x='bedrooms', y='bathrooms', data=hs)
p = hs.hist(figsize = (20,20))
plt.figure()

sns.distplot(hs['bedrooms'])

plt.show()

plt.close()
plt.figure()

sns.distplot(hs['bathrooms'])

plt.show()

plt.close()
sns.kdeplot(data=hs['bedrooms'],label='Bedrooms',shade=True)
sns.kdeplot(data=hs['bathrooms'],label='Bathrooms',shade=True)