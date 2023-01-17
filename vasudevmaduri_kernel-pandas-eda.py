# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import tensorflow as tf

import keras

import matplotlib

import matplotlib.pylab as plt

import seaborn as sns
df = pd.read_csv("../input/googleplaystore.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
df.describe(include=['object'])
for col in df.columns:

    df[col] = df[col].astype('str')
df_col = df.columns

category_distinct = df.Category.unique()
categories = dict(df['Category'].value_counts())



lists = sorted(categories.items()) # sorted by key, return a list of tuples



x, y = zip(*lists) # unpack a list of pairs into two tuples



plt.plot(y, x, linewidth = 2,)

plt.show()

df.apply(np.max)
columns_to_show = ['App', 'Category', 'Installs']

rating_data = df.groupby(['Rating'])[columns_to_show].head()

sorted_data = rating_data.sort_values(by='Installs', ascending = False).head()
pd.crosstab(df['Category'],df['Installs'], normalize = True)