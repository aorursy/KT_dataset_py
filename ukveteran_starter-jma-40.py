from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



el = pd.read_csv('../input/elnino.csv')

el.columns = [col.strip() for col in el.columns]

el.columns = [col.replace(' ','_') for col in el.columns]
el.head()
el.describe()
plt.matshow(el.corr())

plt.colorbar()

plt.show()
p = el.hist(figsize = (20,20))
el['Zonal_Winds'] = pd.to_numeric(el['Zonal_Winds'], errors='coerce')

el['Zonal_Winds'].describe()
el['Meridional_Winds'] = pd.to_numeric(el['Meridional_Winds'], errors='coerce')

el['Meridional_Winds'].describe()
sns.jointplot(x="Zonal_Winds", y="Meridional_Winds", data=el)
el['Air_Temp'] = pd.to_numeric(el['Air_Temp'], errors='coerce')

el['Air_Temp'].describe()
sns.jointplot(x="Zonal_Winds", y="Air_Temp", data=el)
sns.jointplot(x="Meridional_Winds", y="Air_Temp", data=el)
sns.lineplot(x='Zonal_Winds', y='Meridional_Winds', data=el)
sns.lineplot(x='Zonal_Winds', y='Air_Temp', data=el)
sns.lineplot(x='Meridional_Winds', y='Air_Temp', data=el)