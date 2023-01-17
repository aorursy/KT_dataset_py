# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
zomato_df=pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")

zomato_df.head()
zomato_df.drop(['url','address','menu_item','reviews_list'],axis=1,inplace=True)

zomato_df.head()
zomato_df.rename(columns={'approx_cost(for two people)':'Aprox_For_2','listed_in(type)':'Type','listed_in(city)':'City','name':'Name'},inplace=True)

zomato_df.head()
gk=zomato_df.groupby('City')

gk.head()
sns.heatmap(zomato_df.isnull())
zomato_df['rate' and 'phone'].fillna(0,inplace=True)

zomato_df.drop('dish_liked',axis=1,inplace=True)

zomato_df.head()
sns.heatmap(zomato_df.isnull())
zomato_df['Aprox_For_2' and 'rest_type' and 'rate'].fillna(0,inplace=True)

zomato_df['rest_type'].fillna(0,inplace=True)

zomato_df['Aprox_For_2'].fillna(0,inplace=True)
sns.heatmap(zomato_df.isnull())
gk=zomato_df.groupby(['City','Name'])

gk.head()
zomato_df.shape
sns.catplot('rate',data=zomato_df,kind="count")

plt.title('No. of Restaurants with maximum rating')
zomato_df.rate.unique()
zomato_df['rate']=zomato_df['rate'].astype(str)

zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('NEW','NAN'))

zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('-','NAN'))
zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('/5',''))

zomato_df.head()
type_plt=pd.crosstab(zomato_df['rate'],zomato_df['Type'])

type_plt.plot(kind='bar',stacked=True);

plt.title('Type - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Type',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=8,fontweight='bold')

plt.yticks(fontsize=5,fontweight='bold');
zomato_df['Aprox_For_2']=zomato_df['Aprox_For_2'].astype(str)

zomato_df['Aprox_For_2']=zomato_df['Aprox_For_2'].apply(lambda x:x.replace(',',''))

zomato_df.info()
IN_Budget=[]

for Aprox_For_2 in zomato_df.Aprox_For_2:

    if int(Aprox_For_2) <= 800:

        IN_Budget.append('In Budget')

    else:

        IN_Budget.append('Expensive')
zomato_df['In_Budget']=IN_Budget

zomato_df.head(77)
sns.catplot('In_Budget',data=zomato_df,kind="count")

plt.title('No. of in budget reataurants')
sns.catplot('online_order',data=zomato_df,kind='count')

plt.title('Restaurants delivering online or Not')
