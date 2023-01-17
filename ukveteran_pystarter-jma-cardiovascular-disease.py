import os

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math as mt

import scipy



from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
url = '../input/cardiovascular-disease/cardiovascular.txt'

data = pd.read_csv(url,sep=';',decimal=',')



# let's separate index from other columns

data.index = data.iloc[:,0]

df = data.iloc[:,1:]



df = df.drop(['chd','famhist'],axis=1)
data.head()
data.shape
df.head()
df.dtypes
df = df.astype('float')
df.dtypes
df.describe()
sns.pairplot(df)
ax = sns.scatterplot(x="sbp", y="obesity",  data=df)