import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/911calls/911.csv')
df.info()
df.head()
df['zip'].head()
df['twp'].head(5)
len(df['title'].value_counts()) #actual call df['title'].nunique()
#actual call 

df['title'].nunique()
df['Reason for call'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason for call'].value_counts()
df['Reason for call'].value_counts()
sns.countplot(x='Reason for call',data=df)
#type(df['timeStamp']) #actual call type(df['timeStamp'].iloc[0])

s = df['timeStamp'][0]

type(s)
#type(df['timeStamp'])

s = df['timeStamp'][0]

type(s) ##actual call type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp']) ##IMP
time = df['timeStamp'].iloc[0]

time.hour
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek) #also we can add data by .date call
df.head()
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
ns_day_of_week_lookup = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'} #lookup table
df['Day of Week'] = df['Day of Week'].map(ns_day_of_week_lookup)
df['Day of Week'].value_counts()
sns.countplot(x='Day of Week',data=df,hue='Reason for call')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Day of Week',data=df,hue='Reason for call')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=1.)

plt.title('Reason for emergency call')
sns.countplot(x='Month',data=df,hue='Reason for call')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=df,hue='Reason for call')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.ylabel('No. Of Calls')

plt.title('Month Wise Split')
byMonth=df.groupby('Month').count()

byMonth.head()
byMonth['twp'].plot()

plt.ylabel('Count of twp')
plt.figure(figsize=(12,6))

sns.lmplot(x='Month',y='twp',data=byMonth.reset_index(),height=8)

plt.tight_layout()
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
df.groupby('Date').count().head()
plt.figure(figsize=(12,6))

df.groupby('Date').count()['twp'].plot() #imp

plt.tight_layout()
plt.figure(figsize=(12,6))

df[df['Reason for call']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Traffic Fire')

plt.tight_layout()
plt.figure(figsize=(12,6))

df[df['Reason for call']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS Traffic')

plt.tight_layout()
plt.figure(figsize=(12,6))

df[df['Reason for call']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS Traffic')

plt.tight_layout()
df.head()
#dayHour = pd.pivot_table(df,values='',index='Day of Week',columns='Hour')
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason for call'].unstack() #important

dayHour.head()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason for call'].unstack() #important

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason for call'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')
plt.figure(figsize=(12,6))

sns.clustermap(dayMonth,cmap='viridis')