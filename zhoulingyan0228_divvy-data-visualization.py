# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_raw = pd.read_csv('../input/data.csv')
df_sub = df_raw.iloc[np.random.choice(df_raw.shape[0], size=10000)][['year', 'month', 'week', 'day', 'hour', 'usertype', 'gender',
       'tripduration', 'temperature', 'events']]
df = df_sub.copy()
scatter_matrix(df, alpha=0.2, figsize=(32, 32), diagonal='kde')
plt.figure()
pd.get_dummies(df['events']).sum().rename("events distribution").plot(kind='pie', figsize=(8, 8), fontsize=10, autopct='%1.0f%%')
plt.figure()
pd.get_dummies(df['usertype']).sum().rename("usertype distribution").plot(kind='pie', figsize=(8, 8), fontsize=10, autopct='%1.0f%%')
plt.figure()
pd.get_dummies(df['gender']).sum().rename("gender distribution").plot(kind='pie', figsize=(8, 8), fontsize=10, autopct='%1.0f%%')
plt.figure()
df1 = pd.concat([pd.get_dummies(df['events']), df[['hour']]], axis=1)
df1.groupby('hour').sum().plot()

plt.figure()
df1 = pd.concat([pd.get_dummies(df['gender']), df[['hour']]], axis=1)
df1.groupby('hour').sum().plot()
df1 = df[['hour', 'tripduration']]
plt.figure()
df1.groupby('hour').count().plot()
plt.figure()
df1.groupby('hour').mean().plot()
df1 = df[['year', 'hour']]
plt.figure()
df1.groupby(['year', 'hour']).size().to_frame().pivot_table(values=0, index=['hour'],
                                                columns=['year'], aggfunc=np.sum).plot.bar(stacked=True)
df1 = df[['hour', 'gender']]
plt.figure()
df1.groupby(['hour', 'gender']).size().to_frame().pivot_table(values=0, index=['hour'],
                                                columns=['gender'], aggfunc=np.sum).plot.bar(stacked=True)