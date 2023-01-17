



import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_json("../input/train.json")
data.head()
data.describe()
msno.matrix(data,figsize=(13,3))
dataPriceLimited = data.copy()

upperLimit = np.percentile(dataPriceLimited.price.values, 99)

dataPriceLimited['price'].ix[dataPriceLimited['price']>upperLimit] = upperLimit

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(13,5)

sn.distplot(data.price.values, bins=50, kde=True,ax=ax1)

sn.distplot(dataPriceLimited.price.values, bins=50, kde=True,ax=ax2)
fig, (axes) = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(13, 8)

sn.boxplot(data=data,y="price",orient="v",ax=axes[0][0])

sn.boxplot(data=data,y="price",x="interest_level",orient="v",ax=axes[0][1])

sn.boxplot(data=dataPriceLimited,y="price",orient="v",ax=axes[1][0])

sn.boxplot(data=dataPriceLimited,y="price",x="interest_level",orient="v",ax=axes[1][1])
fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(13,5)



interestGroupedData = pd.DataFrame(data.groupby("interest_level")["price"].mean()).reset_index()

interestGroupedSortedData = interestGroupedData.sort_values(by="price",ascending=False)

sn.barplot(data=interestGroupedSortedData,x="interest_level",y="price",ax=ax1,orient="v")

ax1.set(xlabel='Interest Level', ylabel='Average Price',title="Average Price Across Interest Level")



interestData = pd.DataFrame(data.interest_level.value_counts())

interestData["interest_level_original"] = interestData.index

sn.barplot(data=interestData,x="interest_level_original",y="interest_level",ax=ax2,orient="v")

ax2.set(xlabel='Interest Level', ylabel='Interest Level Frequency',title= "Frequency By Interest Level")
fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(13,8)



sn.countplot(x="bathrooms", data=data,ax=ax1)

data1 = data.groupby(['bathrooms', 'interest_level'])['bathrooms'].count().unstack('interest_level').fillna(0)

data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2)

fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(13,8)



sn.countplot(x="bedrooms", data=data,ax=ax1)

data1 = data.groupby(['bedrooms', 'interest_level'])['bedrooms'].count().unstack('interest_level').fillna(0)

data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2)
data["created"] = pd.to_datetime(data["created"])

data["hour"] = data["created"].dt.hour

fig,(ax1,ax2)= plt.subplots(nrows=2)

fig.set_size_inches(13,8)



sn.countplot(x="hour", data=data,ax=ax1)



data1 = data.groupby(['hour', 'interest_level'])['hour'].count().unstack('interest_level').fillna(0)

data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2,)

fig,(ax1)= plt.subplots()

fig.set_size_inches(13,8)

ax1.scatter(data[data['interest_level']=="low"]['bedrooms'],data[data['interest_level']=="low"]['bathrooms'],c='green',s=40)

ax1.scatter(data[data['interest_level']=="medium"]['bedrooms'],data[data['interest_level']=="medium"]['bathrooms'],c='red',s=40)

ax1.scatter(data[data['interest_level']=="high"]['bedrooms'],data[data['interest_level']=="high"]['bathrooms'],c='blue',s=80)

ax1.set_xlabel('Bedrooms')

ax1.set_ylabel('Bathrooms')

ax1.legend(('Low','Medium','High'),scatterpoints=1,loc='upper right',fontsize=15,)
corrMatt = data[["bedrooms","bathrooms","price"]].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)