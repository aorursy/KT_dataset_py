

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df=pd.read_csv('../input/911.csv')

df.info()

df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason']=df['title'].apply(lambda title:title.split(':')[0])
df['Reason'].value_counts(5)
import seaborn as sns

sns.countplot(x='Reason',data=df,palette='viridis')
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df.head()
df['hour']=df['timeStamp'].apply(lambda time:time.hour)

df['month']=df['timeStamp'].apply(lambda time:time.month)

df['day of week']=df['timeStamp'].apply(lambda time:time.dayofweek)

df['day of week']

df.head()
dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day of week']=df['day of week'].map(dmap)

df['day of week']
df.head()
sns.countplot(x='day of week',data=df,palette='viridis',hue='Reason')

plt.legend(bbox_to_anchor=(1.05,1),loc=2)
sns.countplot(x='month',data=df,palette='viridis',hue='Reason')

plt.legend(bbox_to_anchor=(1.05,1),loc=2)
byMonth=df.groupby('month').count()
byMonth.head()
byMonth['lat'].plot()
sns.lmplot(x='month',y='twp',data=byMonth.reset_index())
df['date']=df['timeStamp'].apply(lambda t:t.date())

df['date']
df.groupby('date').count()['lat'].plot()

plt.tight_layout()
dayHour=df.groupby(by=['day of week','hour']).count()['Reason'].unstack()
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth=df.groupby(by=['day of week','month']).count()['Reason'].unstack()

sns.clustermap(dayMonth,cmap='viridis')
sns.heatmap(dayMonth,cmap='viridis')