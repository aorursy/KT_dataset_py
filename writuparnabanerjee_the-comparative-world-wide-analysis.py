# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
data.shape
data.info()
data.isnull().sum()
data.head()
df=data.copy()
df=df.fillna(0)
df.isnull().sum()
cols = df[['Pop. Density (per sq. mi.)' , 'Coastline (coast/area ratio)' , 'Net migration' , 'Infant mortality (per 1000 births)' , 

                   'Literacy (%)' , 'Phones (per 1000)' , 'Arable (%)' , 'Crops (%)' , 'Other (%)' , 'Climate' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,

                   'Industry' , 'Service']]

def rectify(cols):

    for i in cols:

        df[i] = df[i].astype(str)

        new_col = []

        for val in df[i]:

            val = val.replace(',','.')

            val = float(val)

            new_col.append(val)



        

        df[i] = new_col





rectify(cols)

df.info()
df.head()
df
df=df.drop(columns=['Other (%)'])
plt.subplots(figsize=(10,5))

sns.heatmap(df.corr(),linewidth=0.5)

plt.show()
plt.subplots(figsize=(8,8))

df1=df.sort_values('Population',ascending=False).head(10)

plt.pie('Population', labels='Country', autopct="%0.2f%%",data=df1)

plt.show()
plt.subplots(figsize=(12,5))

df4=df.sort_values('Birthrate',ascending=False).head(10)

sns.barplot(x='Country',y='Birthrate',hue='Region',data=df4)

plt.show()
plt.subplots(figsize=(12,5))

df5=df.sort_values('Deathrate',ascending=False).head(10)

sns.barplot(x='Country',y='Deathrate',hue='Region',data=df5)

plt.show()
plt.subplots(figsize=(12,5))

df6=df.sort_values('Literacy (%)',ascending=False).head(10)

sns.barplot(x='Country',y='Literacy (%)',hue='Region',data=df6)

plt.show()
plt.subplots(figsize=(10,8))

df_new=df.sort_values('Population',ascending=False).head(10)

sns.barplot(x='Country', y='Literacy (%)', hue='Population',data=df_new)

plt.show()
plt.subplots(figsize=(12,5))

sns.distplot(df['Birthrate'],hist=False,label='Birthrate')

sns.distplot(df['Deathrate'],hist=False, label='Deathrate')

sns.distplot(df['Infant mortality (per 1000 births)'],hist=False, label='Infant Mortality')

plt.xlabel('Rate')

plt.show()
plt.subplots(figsize=(5,5))

df7=df[df['Deathrate']>df['Birthrate']].shape[0]

plt.pie([df7,(df.shape[0]-df7)],labels=['Death Rate > Birth Rate','Birth Rate > Death Rate'],autopct="%0.2f%%")

plt.show()
plt.subplots(figsize=(12,5))

df3=df.sort_values('Industry',ascending=False).head(10)

sns.barplot(x='Country',y='Industry',hue='Region',data=df3)

plt.show()
plt.subplots(figsize=(15,5))

df2=df.sort_values('GDP ($ per capita)',ascending=False).head(10)

sns.barplot(x='Country',y='GDP ($ per capita)',hue='Region',data=df2)

plt.show()
sns.jointplot(x='Industry',y='GDP ($ per capita)',kind='reg',data=df)

plt.show()

sns.jointplot(x='Service',y='GDP ($ per capita)',kind='reg',color='r',data=df)

plt.show()
sns.jointplot(x='Agriculture',y='GDP ($ per capita)',kind='reg',data=df,color='g')

plt.show()
sns.stripplot(x='Climate',y='Crops (%)',data=df,color='maroon')

plt.show()
plt.subplots(figsize=(12,5))

df9=df.sort_values('Arable (%)',ascending=False).head(10)

sns.barplot(x='Country',y='Arable (%)',hue='Region',data=df9)

plt.show()