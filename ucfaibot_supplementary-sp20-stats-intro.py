import pandas as pd

import numpy as np

from sklearn.datasets import load_boston

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
## Load the Boston dataset into a variable called boston

boston = load_boston()
## Separate the features from the target

x = boston.data

y = boston.target
## Take the columns separately in a variable

columns = boston.feature_names



## Create the Pandas dataframe from the sklearn dataset

boston_df = pd.DataFrame(boston.data)

boston_df.columns = columns
boston_df.describe()
print ("Rows     : " , boston_df.shape[0])

print ("Columns  : " , boston_df.shape[1])

print ("\nFeatures : \n" , boston_df.columns.tolist())

print ("\nMissing values :  ", boston_df.isnull().sum().values.sum())

print ("\nUnique values :  \n",boston_df.nunique())

print('\n')

print(boston_df.head())
rooms = boston_df['RM']

rooms.mean(), rooms.median()
rooms.std(), rooms.var()
sns.distplot(rooms)
stats.normaltest(rooms)
age = boston_df['AGE']

print(age.std(), age.mean())

sns.distplot(age)
log_age = np.log(age)

print(log_age.std(), log_age.mean())

sns.distplot(log_age)
sns.boxplot(x=boston_df['DIS'])

plt.show()
z = np.abs(stats.zscore(boston_df))

print(z)
threshold = 3

## The first array contains the list of row numbers and the second array contains their respective column numbers.

print(np.where(z > 3))
print(boston_df.shape)

boston_df = boston_df[(np.abs(stats.zscore(boston_df)) < 3).all(axis=1)]

print(boston_df.shape)