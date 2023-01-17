import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
df = pd.read_csv('../input/review-data/review_data.csv',index_col=0)

df.head()
df.describe().transpose()
plt.figure(figsize=(10,5))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.count()
date = []

month = []

year = []

rate = []



for i in range(len(df)):

    a = df['Review Date'][i].split(' ')[-3:]

    b = df['Rating(Out of 5)'][i].split(' ')[0]

    date.append(int(a[0]))

    month.append(a[1])

    year.append(int(a[2]))

    rate.append(float(b))

    

date = pd.DataFrame({'Date':date,

                     'Month':month,

                     'Year':year,

                     'Rating(out of 5)':rate})

date.head()
data = df.join(date).drop(['Review Date','Rating(Out of 5)'],axis=1)

data.head()
data.describe().transpose()
plt.figure(figsize=(10,5))

sns.distplot(data['Date'])

plt.figure(figsize=(10,5))

sns.countplot(data['Date'])
plt.figure(figsize=(10,5))

sns.distplot(data['Rating(out of 5)'])

plt.figure(figsize=(10,5))

sns.countplot(data['Rating(out of 5)'])
#Visualizations for year = 2019

a = []

b =[]

c =[]

for i in range(len(data)):

    if data['Year'].iloc[i]==2019:

        a.append(data['Date'].iloc[i])

        b.append(data['Month'].iloc[i])

        c.append(data['Rating(out of 5)'].iloc[i])

df1 = pd.DataFrame({'Date':a,

                    'Month':b,

                    'Rating(out of 5)':c})





plt.figure(figsize=(10,5))

sns.countplot(df1['Month'])



plt.figure(figsize=(10,5))

sns.boxplot(x='Month',y='Date',data=df1)



plt.figure(figsize=(10,5))

sns.boxplot(x='Rating(out of 5)',y='Date',data=df1)



plt.figure(figsize=(10,5))

sns.boxplot(x='Month',y='Rating(out of 5)',data=df1)



plt.figure(figsize=(10,5))

sns.swarmplot(x='Rating(out of 5)',y='Date',data=df1)
#Visualizations for year = 2020

a = []

b =[]

c =[]

for i in range(len(data)):

    if data['Year'].iloc[i]==2020:

        a.append(data['Date'].iloc[i])

        b.append(data['Month'].iloc[i])

        c.append(data['Rating(out of 5)'].iloc[i])

df2 = pd.DataFrame({'Date':a,

                    'Month':b,

                    'Rating(out of 5)':c})





plt.figure(figsize=(10,5))

sns.countplot(df2['Month'])



plt.figure(figsize=(10,5))

sns.boxplot(x='Month',y='Date',data=df2)



plt.figure(figsize=(10,5))

sns.boxplot(x='Rating(out of 5)',y='Date',data=df2)



plt.figure(figsize=(10,5))

sns.boxplot(x='Month',y='Rating(out of 5)',data=df2)



plt.figure(figsize=(10,5))

sns.swarmplot(x='Rating(out of 5)',y='Date',data=df2)
len(df1)
len(df2)
#year = 2019

plt.figure(figsize=(10,5))

sns.countplot(df1['Month'])



plt.figure(figsize=(10,5))

sns.barplot(x='Month',y='Rating(out of 5)',data=df1)
#year = 2020

plt.figure(figsize=(10,5))

sns.countplot(df2['Month'])



plt.figure(figsize=(10,5))

sns.barplot(x='Month',y='Rating(out of 5)',data=df2)
data['Rating(out of 5)'].value_counts()


# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = '5.0', '4.0', '3.0', '2.0', '1.0'

sizes = [568,169,51,30,56]

explode = (0.1, 0.1, 0.1, 0.1,0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
date = []

for i in range(len(data)):

    a = datetime.date(data['Year'].iloc[i],datetime.datetime.strptime(data['Month'].iloc[i],'%B').month,data['Date'].iloc[i])

    date.append(a)



dff = pd.DataFrame({'Reviewed on':date})

new = data.join(dff).drop(['Date','Month','Year'],axis=1)

new
new.to_csv('new_yogabars.csv',index=False)