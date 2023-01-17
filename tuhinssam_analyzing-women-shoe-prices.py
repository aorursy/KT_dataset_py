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
#import Computation Libraries

import pandas as pd

import numpy as np



#import Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns
#read women shoe prices csv

df_women_shoes = pd.read_csv("/kaggle/input/womens-shoes-prices/7210_1.csv")
#check for sample data in the csv

df_women_shoes.sample(10)
#Get information about the column properties, datatypes, notnull count

df_women_shoes.info()
# Here only choose 5 columns interesting for the analysis of the data.

columns = ['brand', 'prices.amountMin', 'prices.amountMax', 'prices.isSale', 'prices.currency']
df_women_shoes = df_women_shoes[columns]

df_women_shoes.shape
df_women_shoes.info()
df_women_shoes.dropna(inplace=True)

df_women_shoes.shape
for currency, subset in df_women_shoes.groupby('prices.currency'):

    print(currency,len(subset))
df_women_shoes[df_women_shoes['prices.currency']!='USD']
for row in df_women_shoes.head().itertuples():

    #print(row._5)

    if row._5 == 'USD':

        pass

    elif row._5 == 'GBP':

        df_women_shoes['prices.amountMin'][row.index] *= 1.3

        df_women_shoes['prices.amountMax'][row.index] *= 1.3

        df_women_shoes['prices.isSale'][row.index] *= 1.3

    elif row._5 == 'EUR':

        df_women_shoes['prices.amountMin'][row.index] *= 1.1

        df_women_shoes['prices.amountMax'][row.index] *= 1.1

        df_women_shoes['prices.isSale'][row.index] *= 1.1

    elif row._5 == 'CAD':

        df_women_shoes['prices.amountMin'][row.index] *= 0.73

        df_women_shoes['prices.amountMax'][row.index] *= 0.73

        df_women_shoes['prices.isSale'][row.index] *= 0.73

    elif row._5 == 'AUD':

        df_women_shoes['prices.amountMin'][row.index] *= 0.75

        df_women_shoes['prices.amountMax'][row.index] *= 0.75

        df_women_shoes['prices.isSale'][row.index] *= 0.75
#create a column for average price

df_women_shoes['prices.average']= (df_women_shoes['prices.amountMin'] + df_women_shoes['prices.amountMax'])/2
#check newly added prices.average column

df_women_shoes.head()
shoelist=[]

for brand,subset in df_women_shoes.groupby('brand'):

    shoelist.append([brand,subset['prices.amountMax'].mean(axis=0)])
#create a new dataframe for analysis

dfshoes= pd.DataFrame(shoelist,columns=['brand','avgprice'])
#sort values by average price

dfshoes.sort_values('avgprice',ascending=False,axis=0,inplace=True)

dfshoes.head(10)
#check the price range for top 10 brands

arr = np.arange(1,11)

plt.figure(figsize=(10,6))

plt.barh(arr,dfshoes['avgprice'].head(10), tick_label=dfshoes['brand'].head(10))

plt.gca().invert_yaxis()

plt.ylabel('Brands', fontsize =12)

plt.isinteractive=True
ax = dfshoes.head(10).plot(kind='barh',figsize=(10,6))

#plt.xlabel('Price in USD')

plt.ylabel('Brands',fontsize=12)

ax.invert_yaxis()

plt.title('Most experience average price brand')

plt.show()
listrange =[]

for brand, subset in df_women_shoes.groupby('brand'):

    listrange.append([brand, abs(subset['prices.average'].max() - subset['prices.average'].min())])

dfpricerange = pd.DataFrame(listrange, columns=['brand','range'])

dfpricerange.head(10)

newdf = dfpricerange.sort_values('range', ascending = False, axis=0).head(10)

newdf

plt.figure(figsize=(10,6))

#plt.xlabel('Price Range (USD)', fontsize=16)

plt.ylabel('Brands', fontsize=16)

plt.barh(np.arange(10),newdf['range'],tick_label=newdf['brand'])

plt.gca().invert_yaxis()

plt.show()
plt.hist(dfshoes['avgprice'],bins=50)

plt.show()
fig,axs = plt.subplots(2,5,figsize=(15,6))



for idx,brand in enumerate(df_women_shoes['brand'].value_counts().sort_values(ascending=False)[0:10].index):

    print(idx,brand)

    axs[idx//5,idx%5].hist(df_women_shoes[df_women_shoes['brand']==brand]['prices.average'], bins=20)

    axs[idx//5,idx%5].set_title(brand)

plt.suptitle("Shoe price pattern")

plt.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()