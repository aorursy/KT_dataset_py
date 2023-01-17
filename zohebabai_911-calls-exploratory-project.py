# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/911.csv")
df.head()
df.info()
# We want top five zip codes for 911 cals
df['zip'].value_counts().head()
# No of unique zip codes
df['zip'].nunique()
#we want top 5 townships for 911 calls
df['twp'].value_counts().head()
# No of unique townships 
df['twp'].nunique()
#how many unique title are there?
df['title'].nunique()
#Create new type and reason column containing reason of title.
df['type'] = df['title'].apply(lambda title : title.split(':')[0])
df['type'].head()
df['reason'] = df['title'].apply(lambda title : title.split(':')[1])
df['reason'].head()
#Most common type of 911 calls
df['type'].value_counts().head(1)
#Countplot of 911 calls by their type
plt.figure(figsize=(10,6))
sns.countplot(x='type',data=df, palette='viridis')
plt.title('Type of Calls')
#Most common reason for 911 calls. We have to remove hyphen or its double counting few reasons.
df['reason'] = df['reason'].apply(lambda t : t.replace('-','').replace(' ',''))
df['reason'].value_counts().head(1)
#Countplot of 911 calls by their top 10 reasons
plt.figure(figsize=(12,6))
sns.countplot( y='reason',data=df,order=df['reason'].value_counts().index[:10],hue='type', palette='viridis')
plt.legend(loc=4)
plt.title('Top 10 Reason of call ')
# Countplot of type of calls based on top 10 township
plt.figure(figsize=(18,6))
sns.countplot( x='twp',data=df,order=df['twp'].value_counts().index[:10], hue='type', palette='viridis')
plt.title('Township wise type of calls')
# We want to see countplot of top 10 Emeregency reasons in Lower Merion township.
plt.figure(figsize=(10,6))
test1=df[df['twp']=='LOWER MERION']
test2=test1[test1['type']=='EMS']
sns.countplot(y='reason',data=test2,order=test2['reason'].value_counts().index[:10],palette='RdYlBu')
plt.title('LOWER MERION EMERGENCY CALLS')
# We want to see countplot of top 5 Traffic reasons in Lower Merion township.
plt.figure(figsize=(12,6))
test3=df[df['twp']=='LOWER MERION']
test4=test3[test3['type']=='Traffic']
sns.countplot(y='reason',data=test4, order=test4['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('LOWER MERION TRAFFIC CALLS')
# We want to see countplot of top 10 Emeregency reasons in Norristown township.
plt.figure(figsize=(10,6))
test5=df[df['twp']=='NORRISTOWN']
test6=test5[test5['type']=='EMS']
sns.countplot(y='reason',data=test6,order=test6['reason'].value_counts().index[:10],palette='RdYlBu')
plt.title('NORRISTOWN EMERGENCY CALLS')
# We want to see countplot of top 5 Traffic reasons in Norristown township.
plt.figure(figsize=(12,6))
test7=df[df['twp']=='NORRISTOWN']
test8=test7[test7['type']=='Traffic']
sns.countplot(y='reason',data=test8, order=test8['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('NORRISTOWN TRAFFIC CALLS')
# We want to see countplot of top 5 Fire reasons in Lower Merion township.
plt.figure(figsize=(12,6))
test9=df[df['twp']=='LOWER MERION']
test10=test9[test9['type']=='Fire']
sns.countplot(y='reason',data=test10, order=test10['reason'].value_counts().index[:5],palette='RdYlBu')
plt.title('LOWER MERION FIRE CALLS')
#We timestamp column is string type. So we will chnage it to datetime object
df['timeStamp']= pd.to_datetime(df['timeStamp'])
#Now we will create three separate columns Hour, Month, Day of Week from this datetime object
df['Hour'] = df['timeStamp'].apply(lambda time : time.hour)
df['DayofWeek'] = df['timeStamp'].apply(lambda time : time.dayofweek)
df['Month'] = df['timeStamp'].apply(lambda time : time.month)
#Now we will convert the day of week to actual day by following dict
dmap= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['DayofWeek']= df['DayofWeek'].map(dmap)
df.head()
#We want plot countplot to see daywise 911 calls based on different type
plt.figure(figsize=(12,6))
sns.countplot(x='DayofWeek',data=df,hue='type',palette='viridis')
#As legend is inside. We will relocate it
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.title('Daywise distribution of Calls')
# We want to see countplot of hourly distribution of traffic 911 calls.
plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['type']=='Traffic'], palette='viridis')
plt.title('Hourwise distribution of Traffic Calls')
plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['reason']=='CARDIACEMERGENCY'], palette='viridis')
plt.title('Hourwise distribution of Cardiac emergency Calls')
plt.figure(figsize=(12,6))
sns.countplot(x='Hour',data=df[df['type']=='Fire'], palette='viridis')
plt.title('Hourwise distribution of Fire Calls')
#We want plot countplot to see monthwise 911 calls based on different type
plt.figure(figsize=(12,6))
sns.countplot(x='Month',data=df,hue='type',palette='viridis')
#As legend is inside. We will relocate it
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.title('Monthwise distribution of Calls')
#Now we will explore by grouping the monthly datas
bymonth = df.groupby('Month').count()
bymonth.head()
bymonth['lat'].plot() 
sns.lmplot(x='Month',y='twp',data=bymonth.reset_index()) 
#Now we shall introduce a new column date from timestamp
t=df['timeStamp'].iloc[0]
df['Date'] = df['timeStamp'].apply(lambda t : t.date())
#Now we shall group by date and plot any column
plt.figure(figsize=(12,6))
df.groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Calls')
df['Date'].value_counts().head(1)
df['Date'].value_counts().tail(1)
plt.figure(figsize=(12,6))
df[df['type']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Traffic Calls')
plt.figure(figsize=(12,6))
plt.figure(figsize=(12,6))
df[df['type']=='EMS'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Emergency Calls')
plt.figure(figsize=(12,6))
df[df['type']=='Fire'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.title('Date wise distribution of Fire Calls')
dayhour = df.groupby(by=['DayofWeek','Hour']).count()['type'].unstack()
#Now lets see it as a heatmap
plt.figure(figsize=(12,6))
sns.heatmap(dayhour,cmap='viridis')
plt.title('Day-hour distribution of 911 Calls')
#Now lets see a clustermap
plt.figure(figsize=(12,6))
sns.clustermap(dayhour,cmap='coolwarm')
plt.title('Day-hour cluster distribution of 911 Calls')
#Now lets do the same thing for months instead of hours
daymonth = df.groupby(by=['DayofWeek','Month']).count()['type'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(daymonth,cmap='viridis')
plt.title('Day-Month distribution of 911 Calls')
plt.figure(figsize=(12,6))
sns.clustermap(daymonth,cmap='coolwarm')
plt.title('Day-month cluster distribution of 911 Calls')