# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/hostel/Hostel.csv')

df.head()
label=df['City'].unique()

size=df.groupby(['City']).size()

print(size)

colors = ['#FF699D', '#F9E253', '#B1EA21',"#33EBFF","#EA33FF"]

plt.figure(figsize=(6,6))

plt.title("Percentage of each hotel in different cities\n",color='blue',fontsize=20)

diagram=plt.pie(size,colors=colors,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
price_city = df.groupby('City').mean()

price1 = price_city['price']

colors1 = ['#FF699D', '#F9E253', '#B1EA21',"#33EBFF","#EA33FF"]

plt.figure(figsize=(6,6))

plt.bar(df['City'].unique(),price_city['price'], color = colors1)

plt.xticks(rotation=-45) 

plt.xlabel('\n\nCity') 

plt.ylabel('Average Price') 

plt.title('Average price per city\n') 

plt.show()





font = {'family': 'Courier New',

        'color':  'darkred',

        'weight': 'bold',

        'size': 25,

        }

df1=df.head(100)

df2=df1.loc[df['City']=='Kyoto']

prices=df2['price'].to_numpy()

plt.figure(figsize=(10,10))

plt.hist(prices, bins=10,color='green',histtype='bar') 

plt.title("No of hotels in given price in kyoto\n",fontdict=font)

plt.xlabel("\nPrices",color='darkred',fontsize=15)

plt.ylabel("No. of rooms",color='darkred',fontsize=15)
