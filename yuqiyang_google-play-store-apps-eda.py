# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
nan_list = ['na','nan','--']
df= pd.read_csv('../input/googleplaystore.csv',na_values = nan_list)
df.sample(10)
df.info()
df.isnull().sum()
df.describe()
#DESCRIBE object and decide which feature should be convert to numerical feature
df.describe(include=['O'])
try:
    df.columns=[str(col).replace(' ','_').lower() for col in df.columns]
except KeyError:
    pass
df.columns.values
# this seems to be a little tricky that one line in this dataset is dislocate,so I drop it.
#I find this error after trying to convert type of 'insdalls'
try:
    non_rowl=df.loc[df['installs']=='Free']
    print(non_rowl)
    df.drop(index =non_rowl.index, inplace=True)
except AttributeError:
    pass
df['reviews']=df['reviews'].apply(lambda x: int(x))
df['installs']=df['installs'].apply(lambda x: int(str(x).replace(',','').replace("+",'')))
df['price']=df['price'].apply(lambda x: float(str(x).replace('$','')))

def unit(x):
    if 'M' in x:
        x = float(x.replace('M',''))
        return x
    elif 'k' in x:
        x = float(x.replace('k',''))
        x = x/1000
        return x
    else:
        x = np.nan
        return x
df['size'] = df['size'].apply(unit)

df.head() 
#fillna with mean and most frequent
mean_rating=df['rating'].mean()
df['rating']=df['rating'].fillna(mean_rating)
mean_size=df['size'].mean
df['size']=df['size'].fillna(mean_size)

fre_type=df['type'].value_counts().index[0]
df['type']=df['type'].fillna(fre_type)
# first a brief view of all the numerical features
sns.heatmap(df[['rating','reviews','installs','size','price']].corr(), annot=True, fmt='.2f', cmap='YlGnBu_r')
sns.pairplot(df,hue = 'type', palette='Set2')
#category and installs vs category and app counts

category_app=df[['category','app']].groupby(['category']).size().reset_index(name = 'count')
category_installs=df[['category','installs']].groupby(['category'],as_index=False).sum()

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ins = category_installs['installs']
cnt = category_app['count']
#lns1 = ax.plot(df['category'], ins, label = 'installs', color = 'g')
lns1 = ax.plot(category_installs['category'], ins, label = 'installs', color = 'g')
lns2 = ax2.plot(category_app['category'], cnt, label = 'count', color = 'r')
ax.legend(loc = 2)
ax2.legend(loc = 1)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
plt.show()
#here we focus on the outliers in each content_rating.
g = sns.catplot(x = 'category', y = 'installs', 
                col = 'content_rating', 
                kind='boxen',
                aspect =2.5, height = 4, col_wrap=2, data = df)
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
#now the data in rating is dispersed. By converting a numarical data into a categrical data would help to make further analysis.
df['rating_range'] = 0
df.loc[df['rating'] <=2.0, 'rating_range']=2
df.loc[(df['rating'] >2.0) & (df['rating'] <=3.0), 'rating_range' ]=3
df.loc[(df['rating'] > 3.0) & (df['rating'] <=4.0), 'rating_range']=4
df.loc[(df['rating']>4.0) & (df['rating'] <=4.5), 'rating_range'] = 4.5
df.loc[(df['rating']>4.5) & (df['rating'] <=5), 'rating_range'] = 5                           
df.rating_range.head()
#features of high rating apps
df_high = df.loc[df['rating_range'] >4]#why df[] not 'column_name'
g =sns.catplot(x = 'rating_range', y = 'installs', data = df_high,kind='swarm',
               hue = 'content_rating',
               palette= 'Set2', height=8,aspect=1.5)
#features of high rating apps
df_high = df.loc[df['rating_range'] >4]#why df[] not 'column_name'
g =sns.catplot(x = 'rating_range', y = 'reviews', data = df_high,hue = 'content_rating',
               palette= 'Set2', height=8,aspect=1.5)
sns.catplot(x = 'content_rating', y = 'rating',hue='type',data=df,
            kind = 'violin', inner = 'stick', split = True,
            height=8, aspect=1.5, palette='Set3')
filter_values = [-1, 0, 5, 10, 20, 50, 100,200, 300, 400]   
df['priceband']= pd.cut(df.price, bins=filter_values)
price_apps= df[['app','priceband']].groupby(['priceband']).size().reset_index(name='appcounts')
price_installs = df[['priceband','installs']].groupby(['priceband'], as_index = False).sum()
pricebands=pd.merge(price_apps, price_installs,on='priceband', how='left')
print(pricebands)
df['price_range'] = 0
df.loc[df['price'] >200, 'price_range']=4
df.loc[(df['price'] >50) & (df['price'] <=200), 'price_range' ]=3
df.loc[(df['price'] > 5) & (df['price'] <=50), 'price_range']=2
df.loc[(df['price']>-1) & (df['price'] <=5), 'price_range'] = 1                         

high_price=df.loc[df['price_range']==4]
plt.figure(figsize = (12,6))
sns.swarmplot(x='category', y='installs', hue='content_rating',size=10, data=high_price, palette='Set2_r')