import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt
dataFrame = pd.read_csv('../input/camera_dataset.csv')
dataFrame.info()

dataFrame=dataFrame.dropna()
dataFrame['BrandName']=dataFrame[['Model']].applymap(lambda x:x.split()[0])

dataFrame['ModelName']=dataFrame[['Model']].applymap(lambda x:x.split()[1])



l=dataFrame['BrandName'].unique()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(l)

dataFrame['BrandName']=le.transform(dataFrame['BrandName'])

dataFrame.sample(10)
dataFrame.info()

dataFrame.describe()
dataFrame['Release date'].value_counts().sort_index().plot(kind='bar',color=['r','g','purple'])

plt.show()
GroupedByDate = dataFrame.groupby(['Release date'])

fig=plt.figure(figsize=(18,6))



ax=plt.subplot2grid((3,5),(0,0),colspan=4,rowspan=2)

GroupedByDate['Max resolution'].max().plot(kind="bar",color='green',legend=True,position=0,width=0.4)

GroupedByDate['Low resolution'].max().plot(kind="bar",color='purple',legend=True,position=1,width=0.4)

plt.show()
fig=plt.figure(figsize=(18,6))

f=dataFrame[dataFrame['Storage included']<128]

plt.subplot2grid((4,5),(0,0),colspan=3,rowspan=2)



plt.scatter(x=f['Release date'], y=f['Storage included'],c='purple',alpha=0.2)

plt.show()
fig=plt.figure(figsize=(18,6))



plt.subplot2grid((3,5),(0,0),colspan=4,rowspan=2)



dataFrame['Storage included'].plot(kind="kde")



plt.show()
fig=plt.figure(figsize=(18,6))



plt.subplot2grid((3,5),(0,0),colspan=4,rowspan=2)

GroupedByDate['Zoom wide (W)'].max().plot(kind="bar",color='green',legend=True,position=0,width=0.4)

GroupedByDate['Zoom tele (T)'].max().plot(kind="bar",color='red',legend=True,position=1,width=0.4)

plt.show()
dataFrame['Dimensions'].plot(kind="kde",color="r")

plt.plot()
fig=plt.figure(figsize=(18,6))

GroupedByDate.max().pivot_table(index='Release date', columns='Price', values='Max resolution').plot.bar(stacked=True)

plt.ylabel("Max resolution")

plt.show()


dataFrameCorr=dataFrame.corr(method='pearson')

plt.subplots(figsize=(8,8))

sns.heatmap(dataFrameCorr, annot=True)

plt.title('Correlation between Attributes')

plt.show()
sns.lmplot(x='Max resolution',y='Low resolution',data=dataFrame,fit_reg=True)

sns.lmplot(x='Max resolution',y='Effective pixels',data=dataFrame,fit_reg=True)

sns.lmplot(x='Low resolution',y='Effective pixels',data=dataFrame,fit_reg=True)

sns.lmplot(x='Zoom wide (W)',y='Weight (inc. batteries)',data=dataFrame,fit_reg=True)

fig=plt.figure(figsize=(18,6))

plt.subplot2grid((5,5),(0,0),rowspan=8,colspan=4)

plt.ylabel("Price")

ax=dataFrame.groupby('BrandName')['Price'].max().plot(kind="bar",color=['r','g','purple','y','black'],position=0,width=0.4,logy=True)

ax=dataFrame.groupby('BrandName')['Price'].min().plot(kind="bar",color=['r','g','purple','y','black'],position=1,width=0.4,logy=True)

ax.set_xticklabels(l,rotation=90)



plt.show()
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((5,5),(0,0),rowspan=8,colspan=4)

ax=sns.boxplot(x='BrandName',y='Price',data=dataFrame)

ax.set_xticklabels(l,rotation=90)

plt.show()
dataFrame.BrandName.unique()
dataFrame.columns
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

dataFrame[['Max resolution','Low resolution']] = min_max_scaler.fit_transform(dataFrame[['Max resolution','Low resolution']])

dataFrame.sample(5)