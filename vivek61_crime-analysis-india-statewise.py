# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/crimeanalysis/crime_by_state.csv')

df.info()
df.describe()
df.isnull().sum()

p=df.groupby(['STATE/UT'])[df.columns.values[2:]].sum()

print(p)
dfs=df.loc[df['STATE/UT']=='TOTAL (ALL-INDIA)']

dfa=dfs.drop(['STATE/UT'],axis=1)

dfa=dfa.melt('Year', var_name='cols',  value_name='vals')

g = sns.factorplot(x="Year", y="vals", hue='cols', data=dfa,size=10,aspect=2)

plt.xlabel('year')

plt.ylabel('count')
ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Murder',ax=ax)
ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Assault on women',ax=ax)

ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Kidnapping and Abduction',ax=ax,color='red')

ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Dacoity',ax=ax,color='green')

ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Robbery',ax=ax,color='blue')
ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Arson',ax=ax,color='yellow')
ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Hurt',ax=ax,color='black')
ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Prevention of atrocities (POA) Act',ax=ax)

ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Protection of Civil Rights (PCR) Act',ax=ax)

ax=plt.gca()

dfs.plot(kind='line',x='Year',y='Other Crimes Against SCs',ax=ax)
