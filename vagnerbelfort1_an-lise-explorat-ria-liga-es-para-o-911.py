import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/911.csv")
df.info()
df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Title2'] = df['title'].apply(lambda title: title.split(':')[1])
df['Reason'].value_counts()
df['Title2'].value_counts().head()
sns.countplot(x='Reason',data=df,palette='viridis')
df['twp'].value_counts().head(10).plot.bar()
plt.figure(figsize=(18,6))

sns.countplot( x='twp',data=df,order=df['twp'].value_counts().index[:10], hue='Reason')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
plt.figure(figsize=(18,5))

sns.countplot(x='Day of Week',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.10, 1))
plt.figure(figsize=(18,6))

sns.countplot(x='Month',data=df,hue='Reason')

plt.legend(bbox_to_anchor=(1.10, 1))