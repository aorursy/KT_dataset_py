# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import numpy as np

import itertools

plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/sweden-covid19-dataset/time_series_confimed-confirmed.csv')

df.head()
df1=pd.read_csv('../input/sweden-covid19-dataset/time_series_deaths-deaths.csv')

df1.head()
plt.figure(figsize=(12,8)) # Figure size

df.groupby("2020-03-24")['Hospital_Total'].max().plot(kind='bar', color='olivedrab')
plt.figure(figsize=(12,8)) # Figure size

df.groupby("2020-03-24")['Hospital_Total'].size().plot(kind='bar', color='olivedrab')
plt.figure(figsize=(12,8)) # Figure size

df.groupby("Today")['Hospital_Total'].max().plot(kind='bar', color='olivedrab')
plt.figure(figsize=(12,8)) # Figure size

df.groupby("Today")['Hospital_Total'].size().plot(kind='bar', color='olivedrab')
plt.figure(figsize=(12,8)) # Figure size

df1.groupby("Today")['FHM_Deaths_Today'].max().plot(kind='bar', color='olivedrab')
plt.figure(figsize=(12,8)) # Figure size

df1.groupby("Today")['FHM_Deaths_Today'].size().plot(kind='bar', color='olivedrab')