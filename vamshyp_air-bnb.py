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
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
label=df['room_type'].unique()

size=df.groupby(['room_type']).size()

print(size)

colors = ['steelblue', 'lightblue', 'darkslateblue']

plt.figure(figsize=(6,6))

plt.title("Percentage of each type of rooms avaiblable\n",color='blue',fontsize=20)

diagram=plt.pie(size,colors=colors,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
price_room = df.groupby('room_type').mean()

price1 = price_room['price']

colors1 = ['darkgray', 'lightslategrey', 'lightsteelblue']

plt.figure(figsize=(6,6))

plt.bar(df['room_type'].unique(),price_room['price'], color = colors1)

plt.xticks(rotation=-45) 

plt.xlabel('\n\nRoom type') 

plt.ylabel('Average Price') 

plt.title('Average price per room type\n') 

plt.show()



# No. of listings in each neighbourhood

flat = ['rosybrown','firebrick','lightcoral','maroon','lightsalmon']

sns.countplot(x='neighbourhood_group', data=df, palette=sns.color_palette(flat))

plt.title('No of Listings in each Neighbourhood\n',fontsize=15,color='darkred') 

# Rotate x-labels

plt.xticks(rotation=-45)

plt.xlabel('Nighbourhood')

plt.ylabel('Listings')
n = df.groupby('neighbourhood_group').mean()

u=df['neighbourhood_group'].count()



n1 = n['price']

colors1 = ['darkgray', 'lightslategrey','dimgrey', 'gainsboro', 'slategrey']

plt.figure(figsize=(10,10))

plt.bar(df['neighbourhood_group'].unique(),n['price'], color = colors1)

plt.xticks(rotation=-45) 

plt.xlabel('\n\nneighbourhood') 

plt.ylabel('Average Price') 

plt.title('Average price per neigbourhood\n') 

plt.show()



font = {'family': 'Courier New',

        'color':  'darkred',

        'weight': 'bold',

        'size': 25,

        }

df1=df.head(100)

df2=df1.loc[df['neighbourhood_group']=='Manhattan']

prices=df2['price'].to_numpy()

plt.figure(figsize=(10,10))

plt.hist(prices, bins=10,color='crimson',histtype='bar') 

plt.title("No of rooms in given price range in Manhattan\n",fontdict=font)

plt.xlabel("\nPrices",color='darkred',fontsize=15)

plt.ylabel("No. of rooms",color='darkred',fontsize=15)
