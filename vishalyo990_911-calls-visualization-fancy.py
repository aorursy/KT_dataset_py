import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#let's import the data.

df = pd.read_csv('../input/911.csv')
#lets look at the dataset.

df.info()

df.head()
#Lets see top 5 zip and townships that have called the 911.

df['zip'].value_counts().head(5)

df['twp'].value_counts().head(5)
#lets see how many unique titles are there in the datasets

df['title'].nunique()
#create a ner column named as 'reason'.

#'reason' column defines the reason for which a call was made.

df['reason'] = df['title'].apply(lambda title: title.split(':')[0])

df['reason'].head(4)

df['reason'].value_counts()
#Plotting a graph for the "reason" variable

sns.countplot(x = 'reason', data = df)
#lets look at the time data in the dataset.

#Convert the time data set into datetime format.

type(df['timeStamp'].iloc[0])

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['timeStamp'].iloc[0]
#Grabbing the date from this timestamp.

df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)

df['Hour'].head(5)
#Now doing the same for the months and day of weeks.

df['month'] = df['timeStamp'].apply(lambda time:time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)

df.head(1)
#Now lets make the day of week as string and provide suitable abbreviations for each week to make it more readiable.

dmap = {0:'mon', 1:'tue',2:'wed', 3:'thu', 4:'fri', 5:'sat', 6:'sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)

df.head(1)
#Lets see the count of the reason of calls per day of week.

sns.countplot(data = df, x = 'Day of Week', hue = 'reason', palette = 'viridis')

plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0)
#Lets do same Visualisations for months

sns.countplot(data = df, x = 'month', hue = 'reason', palette = 'viridis')

plt.legend(bbox_to_anchor = (1.05,1), loc = 2, borderaxespad = 0)

#Now grab the date from the timestamp coloumn

df['date'] = df['timeStamp'].apply(lambda t:t.date())

#Group by the data as per date and plot it.

fig = plt.figure(figsize=(10,5))

df.groupby('date').count()['lat'].plot()

plt.tight_layout()
#now lets plot all the calls for different-different reasons.

#For Traffic

fig = plt.figure(figsize=(10,5))

df[df['reason']=='Traffic'].groupby('date').count()['lat'].plot()

plt.tight_layout()

plt.title('Traffic')
#For EMS

fig = plt.figure(figsize=(10,5))

df[df['reason']=='EMS'].groupby('date').count()['lat'].plot()

plt.tight_layout()

plt.title('EMS')
#For FIRE

fig = plt.figure(figsize=(10,5))

df[df['reason']=='Fire'].groupby('date').count()['lat'].plot()

plt.tight_layout()

plt.title('Fire')
#Now let's do plotting via heatmaps and clustermaps.

#First we need to create the matrix form of data.

dayHour = df.groupby(by = ['Day of Week', 'Hour']).count()['reason'].unstack()

dayHour
#plotting the heatmap

fig = plt.figure(figsize = (10,7))

sns.heatmap(dayHour, cmap = 'viridis')
#Clustermap for the same

fig = plt.figure(figsize = (10,7))

sns.clustermap(dayHour, cmap = 'coolwarm')
#Lets do the same for month and day of week.

#heatmap

fig = plt.figure(figsize = (10,7))

dayMonth = df.groupby(by = ['Day of Week', 'month']).count()['reason'].unstack()

sns.heatmap(dayMonth, cmap = 'viridis')
#Clustermap

fig = plt.figure(figsize = (10,7))

sns.clustermap(dayMonth, cmap = 'coolwarm')