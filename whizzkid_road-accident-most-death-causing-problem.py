import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))
df=pd.read_csv("../input/Traffic accidents by time of occurrence 2001-2014.csv")

df.columns=["state",'year','type','latenight_0-3hrs','earlymorning_3-6hrs','morning_6-9hrs',

            'earlynoon_9-12hrs','noon_12-15hrs','evening_15-18hrs','earlynight_18-21hrs',

           'night_21-24hrs','total']

df.head(2)
def plotbar(df,column,vertical,plottype='bar'):

    count=[]

    for x in df[column].unique():

        c=df[df[column]==x].total.sum()

        count.append(c)

    

    plt.figure(figsize=(10,10))

    if(plottype=='bar'):

        sns.barplot(df[column].unique(),count)

    if(plottype=='line'):

        sns.lineplot(df[column].unique(),count)

    if(vertical):

        plt.xticks(range(df[column].nunique()),rotation='vertical')

    plt.show()
plotbar(df,"year",False,'line')
plotbar(df,'state',True)
plotbar(df,"type",False)
time_column=df.columns[3:-1].values

count=[]

for col in time_column:

    count.append(df[col].sum())

sns.barplot(time_column,count)

plt.xticks(range(len(time_column)),rotation='vertical')

plt.show()
df.groupby("state").sum().iloc[:,1:-1].plot.bar(figsize=(15,10),stacked=True,

                                                title="statewise accidents in each time interval")
df.groupby('year').sum().iloc[:,0:-1].plot(figsize=(15,10),title="Number of accidents per year in each time interval")

plt.show()
df.head()
plt.figure(figsize=(15,10))

sns.lineplot(x="state",y="total",hue="year",data=df)

plt.xticks(rotation='vertical')

plt.show()