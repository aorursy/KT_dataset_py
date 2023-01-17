import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)  # visualization tool





from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

type(df)
df.head()
df.tail()
df.shape
df.describe()
df.info()
df.columns
df.dtypes
df.corr()
df.plot(subplots=True,figsize=(18,18))

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")

plt.show()
sns.pairplot(df.iloc[:,0:8],hue="Country")

plt.show()
fig=plt.figure(figsize=(20,15))

ax=fig.gca()

df.hist(ax=ax)

plt.show()