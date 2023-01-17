from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



diab=pd.read_csv('../input/diabetes.csv')
diab.head()
diab.describe()
plt.matshow(diab.corr())

plt.colorbar()

plt.show()
corr_mat = diab.corr(method='pearson')

plt.figure(figsize=(20,10))

sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
sns.boxplot(diab['Glucose'])
plt.scatter(diab['Glucose'], diab['BloodPressure'])
plt.scatter(diab['BMI'], diab['Age'])
p = diab.hist(figsize = (20,20))
ax = sns.violinplot(x="Glucose", y="Age", data=diab)
ax = sns.violinplot(x=diab["Glucose"])
ax = sns.violinplot(x=diab["Age"])
ax = sns.violinplot(x=diab["BMI"])
ax = sns.violinplot(x=diab["Insulin"])
ax = sns.violinplot(x="Insulin", y="Glucose", data=diab)
ax = sns.violinplot(x="Glucose", y="Insulin", data=diab)