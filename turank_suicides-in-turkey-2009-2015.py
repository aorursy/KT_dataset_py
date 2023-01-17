# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.info()
df.head()
df.describe()
f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(df.corr(),annot=True, lw=.5,fmt='.2f',ax=ax)

plt.show()
df["country"].unique() #All countries that we have in our data.

df = df[df["country"] == 'Turkey'] #Filtering for Turkey

df.head()
df.describe() #Turkey's numerical data information.
#correlation map for Turkey

f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(df.corr(),annot=True, lw=.5,fmt='.2f',ax=ax)

plt.show()

#scatter plot

#year and suicide number



df.plot(kind='scatter',x='year',y='suicides_no')

plt.xlabel('year')

plt.ylabel('suicide no')

plt.show()
#Lets look at maximum suicide number per year and visualize it.

max_suicides= df.groupby(['year']).suicides_no.max() #max suicide numbers

print(max_suicides)

f=plt.subplots(figsize=(10,5))

plt.bar(df['year'].unique(),max_suicides)

plt.xlabel('year')

plt.ylabel('suicide number')

plt.title('maximum number of suicides per year')

plt.show()
#finding min suicide numbers and its bar graph

min_suicides = df.groupby(['year']).suicides_no.min()



plt.figure(figsize=(10,5))

plt.bar(df.year.unique(),min_suicides)

plt.xlabel('year')

plt.ylabel('suicide number')

plt.title('minimum number of suicides per year')



plt.show()



#now we are going to look at total suicide numbers per year

total_suicides = df.groupby(['year']).suicides_no.sum()

total_suicides
#lets visualize it.

#line graph

plt.figure(figsize=(10,10))

plt.plot(df['year'].unique(),total_suicides,lw=2,marker='o')

plt.xlabel('year')

plt.ylabel('total suicide no')

plt.title('relationship between year and total suicide numbers -line graph')

plt.show()
#bar graph

plt.figure(figsize=(10,5))

plt.bar(df['year'].unique(),total_suicides)

plt.xlabel('year')

plt.ylabel('total suicide no')

plt.title('relationship between year and total suicide numbers -bar graph')

plt.show()
plt.figure(figsize=(13,5))

sns.barplot(df.sex,df.suicides_no,hue=df.age)

plt.title('Suicide number by Gender')

plt.show()
#now we are going to look at the relationship between GDP per capita and suicide number

plt.figure(figsize=(10,8))

gdp_suicides = df.groupby(['gdp_per_capita ($)']).suicides_no.sum() #total suicide numbers for each gdp value.

sns.lineplot(df['gdp_per_capita ($)'].unique(),gdp_suicides)

plt.xlabel('gdp per capita ($)')

plt.ylabel('suicides no')

plt.title('relationship between GDP per capita and suicide number')

plt.show()