import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input"))
data=pd.read_csv('../input/zomato.csv')

data.head()
data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',

                         'listed_in(city)':'city'})
data.info()
round((data.isnull().sum()/data.shape[0])*100,2)
data.describe()
url=data.pop('url')

address=data.pop('address')

phone=data.pop('phone')

menu_item=data.pop('menu_item')

reviews_list=data.pop('reviews_list')

type_hotel=data.pop('type')
data['online_order']=data['online_order'].apply(lambda x: '1' if str(x)=='Yes' else '0')

data['book_table']=data['book_table'].apply(lambda x: '1' if str(x)=='Yes' else '0')

data['rate']=data['rate'].apply(lambda x: str(x).split('/')[0])

data['cost']=data['cost'].apply(lambda x: str(x).replace(',',''))

data.dropna(subset=['rate','cost'])

data=data[data['rate']!='nan']

data=data[data['rate']!='NEW']

data=data[data['rate']!='-']

data=data[data['cost']!='nan']

data['rate']=data['rate'].astype(float)

data['votes']=data['votes'].astype(int)

data['cost']=data['cost'].astype(int)
plt.subplots(3,2,figsize=(10,10))

plt.subplot(3,2,1)

sns.countplot(data['online_order'])

plt.subplot(3,2,2)

sns.countplot(data['book_table'])

plt.subplot(3,2,3)

sns.distplot(data['rate'],kde=True)

plt.subplot(3,2,4)

sns.distplot(data['votes'],kde=True)

plt.subplot(3,2,5)

sns.distplot(data['cost'])



plt.tight_layout()
data.sample(5)
plt.figure(figsize=(10,10))

sns.countplot(data['rate'])

data[data['rate']==4.9][:5]
data[data['rate']==1.8][:5]
hotel_counts=data['name'].value_counts()

unique_hotels=data['name'].unique()
hotel_counts
data[data['name']=='Onesta'].head(10)
data.drop_duplicates(keep='first',inplace=True)
data[data['name']=='KFC'].head(10)
data['name'].value_counts()
plt.subplots(3,2,figsize=(10,10))

plt.subplot(3,2,1)

sns.countplot(data['online_order'])

plt.subplot(3,2,2)

sns.countplot(data['book_table'])

plt.subplot(3,2,3)

sns.distplot(data['rate'],kde=True)

plt.subplot(3,2,4)

sns.distplot(data['votes'],kde=True)

plt.subplot(3,2,5)

sns.distplot(data['cost'])



plt.tight_layout()
plt.figure(figsize=(15,8))

sns.countplot(data['rate'],hue='online_order',data=data)
plt.figure(figsize=(15,8))

sns.countplot(data['rate'],hue='book_table',data=data)
plt.figure(figsize=(6,6))

data.groupby('online_order')['rate'].mean().plot.bar()

plt.ylabel('Average rating')
plt.figure(figsize=(6,6))

data.groupby('book_table')['rate'].mean().plot.bar()

plt.ylabel('Average rating')


sns.lmplot(x='votes',y='rate',data=data)



sns.jointplot(x='votes',y='rate',data=data,kind='hex',gridsize=15,color='orange')



sns.lmplot(x='cost',y='rate',data=data)



sns.jointplot(x='cost',y='rate',data=data,color='red',kind='hex',gridsize=15)

plt.figure(figsize=(10,6))

data.groupby('location')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()
top_locations=data[data['location'].isin(['Lavelle Road','Koramangala 3rd Block','Koramangala 5th block',

                                         'St. Marks Road','Sankey Road'])]

sns.violinplot(x='location',y='rate',data=top_locations)

plt.xticks(rotation=90)
plt.figure(figsize=(10,6))

data.groupby('city')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()
top_cities=data[data['city'].isin(['MG Road','Brigade Road','Koramangala 4th block',

                                       'Lavelle Road','Koramangala 7th Block'])]

                                         

sns.violinplot(x='city',y='rate',data=top_cities)

plt.xticks(rotation=90)
plt.figure(figsize=(10,6))

data.groupby('rest_type')['rate'].mean().sort_values(ascending=False)[:10].plot.bar()
data['afford']=data['rate']/data['cost']

data.groupby('location')['afford'].mean().sort_values(ascending=False)[:10].plot.bar()

plt.ylabel('Affordability')
top_afford=data[data['location'].isin(['Basavanagudi','City Market','Commercial Street',

                                      'Shivajinagar','Vijay Nagar'])]

sns.violinplot(x='location',y='afford',data=top_afford)

plt.xticks(rotation=90)
data.sample(5)
data['rest_type'].unique()
dining=data[data['rest_type'].isin(['Casual Dining','Fine Dining'])]

dining.groupby('location')['afford'].mean().sort_values(ascending=False)[:10].plot.bar()

top_afford_dining=dining[dining['location'].isin(['City Market','Uttarahalli','Jalahalli',

                                                 'Mysore Road','KR Puram'])]

sns.violinplot(x='location',y='afford',data=top_afford_dining)

plt.xticks(rotation=90)