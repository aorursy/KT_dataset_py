from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/home-office-airpi/AirPi Data - AirPi.csv')
dat.head()
dat.describe()
p = dat.hist(figsize = (20,20))
plt.matshow(dat.corr())

plt.colorbar()

plt.show()
sns.regplot(x=dat['Volume [mV]'], y=dat['Light_Level [Ohms]'])
sns.regplot(x=dat['Carbon_Monoxide [Ohms]'], y=dat['Nitrogen_Dioxide [Ohms]'])
sns.regplot(x=dat['Temperature-DHT [Celsius]'], y=dat['Temperature-BMP [Celsius]'])