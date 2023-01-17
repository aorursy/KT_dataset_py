import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mlxtend.plotting import category_scatter

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.shape

df.head()
df.info()
df.isnull().sum().sort_values(ascending = False)
df.drop(['host_name','last_review','name'],axis = 1, inplace = True)
df[df['number_of_reviews']==0].shape
df['reviews_per_month'].fillna(0, inplace = True)
df.isnull().sum().sort_values(ascending = False)
df['neighbourhood_group'].unique()
df['room_type'].unique()
n_group = df.groupby('neighbourhood_group').describe()

n_group.xs('price',axis = 1)
sns.catplot(x = 'neighbourhood_group', y = 'price', data = df)
df1 =df[df['price']<500]

plt.figure(figsize = (10,5))

sns.violinplot(x = 'neighbourhood_group', y = 'price', data = df1, scale = 'count', linewidth = 0.3)
plt.figure(figsize=(8,6))

sns.countplot(df['neighbourhood_group'],hue=df['room_type'])
plt.figure(figsize=(8, 6))

sns.barplot(data=df, x='room_type', y='price')
df2= df1.groupby('neighbourhood_group')

fig, ax = plt.subplots(2,3,figsize =(20,10))

ax = ax.flatten().T

sns.scatterplot('number_of_reviews','price',data = df2.get_group('Brooklyn'),ax = ax[0], label ='Brooklyn')

sns.scatterplot('number_of_reviews','price',data = df2.get_group('Manhattan'),ax = ax[1],color = 'orange',label ='Manhattan')

sns.scatterplot('number_of_reviews','price',data = df2.get_group('Bronx'),ax = ax[2],color = 'purple',label ='Bronx')

sns.scatterplot('number_of_reviews','price',data = df2.get_group('Queens'),ax = ax[3],color = 'g',label ='Queens')

sns.scatterplot('number_of_reviews','price',data = df2.get_group('Staten Island'),ax = ax[4],color = 'r',label ='Staten Island')
plt.figure(figsize=(8, 6))

sns.countplot('minimum_nights', data = df1)

plt.xlim(0, 40)

tick = [1,5,10,15,20,25,30,35,40]

plt.xticks(tick, tick)
plt.figure(figsize= (10,8))

plt.scatter(df1.longitude, df1.latitude, c = df1.price, alpha = 0.7, cmap ='jet',edgecolor = 'black')

cbar = plt.colorbar()

cbar.set_label('Price')
plt.figure(figsize= (10,8))

plt.hist2d(df1.longitude, df1.latitude, bins=(100,100),cmap =plt.cm.jet)

c_bar = plt.colorbar()

c_bar.set_label('Density')
plt.figure(figsize= (10,8))

plt.scatter(df1.longitude, df1.latitude, c = df1.availability_365, alpha = 0.7,cmap ='summer',edgecolor = 'black')

c_bar = plt.colorbar()

c_bar.set_label('Availability')