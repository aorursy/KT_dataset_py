import pandas as pd

import numpy as np
data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")



data.head(10)
data.describe()
data.isnull().sum()
total_cells = np.product(data.shape)

total_missing = data.isnull().sum().sum()



(total_missing/total_cells)*100
data = data.fillna(method='bfill').fillna(0,axis=0)
data.isnull().sum()
data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")



data.head(5)
data['airEPA'].isnull().sum()

data['EPA'].isnull().sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer()

df = imputer.fit_transform(data.loc[:,['airEPA','EPA']])
data['airEPA'] = df[:,0]

data['EPA'] = df[:,1]
data.head(3)
data['EPA'].isnull().sum()

data['airEPA'].isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

data = np.random.exponential(size = 1000)



fig,ax = plt.subplots()

ax=sns.distplot(data)

plt.show()
from sklearn.preprocessing import minmax_scale



scaled_data = minmax_scale(data)



scaled_data



fig,ax = plt.subplots()

ax = sns.distplot(scaled_data)

plt.show()
print(data[0:10])

print(scaled_data[0:10])
from sklearn.preprocessing import normalize

from scipy import stats













data = np.random.exponential(size = 1000)







normalize_data = stats.boxcox(data)



normalize_data = normalize_data[0]







fig,ax = plt.subplots()

ax = sns.distplot(normalize_data)

plt.show()



type(normalize_data)








