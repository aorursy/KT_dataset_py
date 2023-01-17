import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
os.chdir("../input")
data =pd.read_csv("vgsales.csv")
data.shape
data.head()
data.columns
data.tail()
data.describe()
plt.figure(figsize=(12,12))

sns.heatmap(data.corr(),annot=True,cmap='RdBu_r')
df1 = data.groupby(['Year'])

plt.figure(figsize=(10,10))

df1_mean = df1['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)

df1_mean.plot(figsize=(10,10))

plt.title('Average sales over the course of years')
plt.figure(figsize=(10,10))

plt.hist(data.Year.values,bins=20)

plt.xlabel('Year')

plt.ylabel('frequency')
data['Genre'].unique()
df3 = data.groupby(['Genre'])

val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)

val.plot(kind='bar',figsize=(20,8))

plt.xlabel('Genre',fontsize=16)

plt.ylabel('Sale of games in each region',fontsize=16)

plt.title('Sales as per Genre',fontsize=16)
data['Platform'].unique()
df3 = data.groupby(['Platform'])



val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)

val.plot(kind='bar',figsize=(20,8))

plt.xlabel('Platform',fontsize=16)

plt.ylabel('Sale of games in each region',fontsize=16)

plt.title('Sales as per Platform',fontsize=16)
df3 = data.groupby(['Platform'])

val = df3['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].aggregate(np.mean)

plt.figure(figsize=(12,10))

ax = sns.boxplot(data=val, orient='h')

plt.xlabel('Revenue per game',fontsize=16)

plt.ylabel('Region',fontsize=16)

plt.title('Distribution of sales as per Platform',fontsize=16)

data.Year.max()
plt.figure(figsize=(12,8))

sns.countplot(x='Genre',data=data)

plt.xlabel('Genre',fontsize=16)

plt.ylabel('Count',fontsize=16)

plt.show()
plt.figure(figsize=(12,8))

sns.barplot(x='Genre',y='Global_Sales',data=data)

plt.xlabel('Genre',fontsize=16)

plt.ylabel('Global Sales',fontsize=16)

plt.title('Global sales as per Genre',fontsize=16)

plt.show()
sns.jointplot(x='NA_Sales',y='Global_Sales',data=data)
plt.figure(figsize=(20,8))

sns.barplot(x='Year',y='Global_Sales',data=data)

plt.title('Global sales per year')

plt.xticks(rotation=45)

plt.show()
df_publishers = data.groupby('Publisher')

plot_publishers = df_publishers['NA_Sales','JP_Sales','EU_Sales','Other_Sales'].mean()

plt.figure(figsize=(12,8))

plot_publishers.boxplot()
sort_publishers = plot_publishers.sort_values('EU_Sales',ascending=False)

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = 'European Union', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).EU_Sales)

plt.title('European Union')

plt.ylabel('Revenue')

plt.suptitle('Revenues per region',size=22)

sort_publishers = plot_publishers.sort_values('NA_Sales',ascending=False)

ax2 = fig.add_subplot(1,4,2,sharey=ax1)

ax2.set_xticklabels(labels = 'North America', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).NA_Sales)

plt.title('North America')

plt.ylabel('Revenue')

sort_publishers = plot_publishers.sort_values('JP_Sales',ascending=False)

ax3 = fig.add_subplot(1,4,3,sharey=ax1)

ax3.set_xticklabels(labels = 'Japan', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).JP_Sales)

plt.title('Japan')

plt.ylabel('Revenue')

sort_publishers = plot_publishers.sort_values('Other_Sales',ascending=False)

ax4 = fig.add_subplot(1,4,4,sharey=ax1)

ax4.set_xticklabels(labels = 'Japan', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).Other_Sales)

plt.title('Other Regions')

plt.ylabel('Revenue')