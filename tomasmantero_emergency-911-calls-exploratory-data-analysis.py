# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/montcoalert/911.csv')
print(df.columns.values)
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])
# preview the data

df.head()
df = df.drop('e',axis=1)
# missing values

print('Missing values:',df.isnull().values.sum())



df.isnull().sum()
df.info()
df['zip'].nunique()
df_zip = pd.DataFrame(df['zip'].value_counts().head(5))

df_zip.rename(columns = {'zip':'Top 5'}, inplace = True)

df_zip.style.background_gradient(cmap='Blues')
df_twp = pd.DataFrame(df['twp'].value_counts().head(5))

df_twp.rename(columns = {'twp':'Top 5'}, inplace = True)

df_twp.style.background_gradient(cmap='Greens')
df['title'].nunique()
df['reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['title_code'] = df['title'].apply(lambda title: title.split(':')[1])
df['reason'].value_counts()
fig, axes = plt.subplots(1,2, figsize=(15, 5))



sns.countplot(x='reason', data=df, order=df['reason'].value_counts().index, ax=axes[0])

axes[0].set_title('Common Reasons for 911 Calls', size=13)

axes[0].set(xlabel='Reason', ylabel='Count')



df['reason'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[1],shadow=True)

axes[1].set(xlabel='', ylabel='')



sns.despine(bottom=False, left=True)
fig, axes = plt.subplots(figsize=(10, 5))

sns.countplot(y='title', data=df, order=df['title'].value_counts().index, palette='prism')

sns.despine(bottom=False, left=True)

axes.set_ylim([9, 0])

axes.set_title('Overall 911 Emregency Calls', size=15)

axes.set(xlabel='Number of 911 Calls', ylabel='')

plt.tight_layout()
df[df['reason']=='Traffic'].groupby('title_code').count()['lat'].sort_values(ascending=True).plot(kind='barh', figsize=(10, 5), color='darkblue')

plt.xlabel('Number of 911 Calls')

plt.ylabel('')

plt.title('Traffic 911 Emergency Calls', fontsize=15)
df[df['reason']=='Fire'].groupby('title_code').count()['lat'].sort_values(ascending=True).tail(10).plot(kind='barh', figsize=(10, 5), color='darkred')

plt.xlabel('Number of 911 Calls')

plt.ylabel('')

plt.title('Fire 911 Emergency Calls', fontsize=15)
df[df['reason']=='EMS'].groupby('title_code').count()['lat'].sort_values(ascending=True).tail(10).plot(kind='barh', figsize=(10, 5), color='darkgreen')

plt.xlabel('Number of 911 Calls')

plt.ylabel('')

plt.title('EMS 911 Emergency Calls', fontsize=15)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])



df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
# dictionary string names

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}



df['Day of Week'] = df['Day of Week'].map(dmap)
fig, axes = plt.subplots(1,2, figsize=(15,5))



sns.countplot(x='Day of Week', data=df, palette='viridis', ax=axes[0])

axes[0].set_title('Weekly Calls', size=15)



sns.countplot(x='Month', data=df, hue='reason', palette='viridis', ax=axes[1])

axes[1].set_title('Monthly Calls', size=15)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



sns.despine(bottom=False, left=True)
df['Date'] = df['timeStamp'].apply(lambda t: t.date())
df[df['reason']=='Traffic'].groupby('Date').count()['lat'].plot(figsize=(15,5), color='darkblue')

plt.title('Traffic', fontsize=15)

sns.despine(bottom=False, left=True)

plt.tight_layout()
df[df['reason']=='Fire'].groupby('Date').count()['lat'].plot(figsize=(15,5), color='darkred')

plt.title('Fire', fontsize=15)

sns.despine(bottom=False, left=True)

plt.tight_layout()
df[df['reason']=='EMS'].groupby('Date').count()['lat'].plot(figsize=(15,5), color='darkgreen')

plt.title('EMS', fontsize=15)

sns.despine(bottom=False, left=True)

plt.tight_layout()
dayHour = df.groupby(by=['Day of Week', 'Hour']).count()['reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour, cmap='viridis', linewidths=0.05)
plt.figure(figsize=(12,6))

sns.clustermap(dayHour, cmap='viridis', linewidths=0.05)