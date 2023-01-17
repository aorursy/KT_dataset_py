# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def dataRead():

    #Read data

    df = pd.read_csv("/kaggle/input/top-250-football-transfers-from-2000-to-2018/top250-00-19.csv")

    return df
df= dataRead()
df.head()

df.info()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (9, 9)

# Histogram

df.hist()

plt.tight_layout()
sns.countplot(x='Position',data=df)

plt.xticks(rotation=90)
sns.pointplot(x='index',y='Season',data=pd.DataFrame(df.Season.value_counts()).reset_index().sort_values('index'))

plt.xticks(rotation=90)

plt.xlabel('Season')

plt.ylabel('Number of Trade')
dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)

dictionary['france'] = "paris"       # Add new entry

print(dictionary)

del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
print(dictionary)       # it gives error because dictionary is deleted
df.info()
series = df['Age']        # data['Defense'] = series

print(type(series))

data_frame = df[['Age']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = df['Age']>30     # There are only 3 pokemons who have higher defense value than 200

df[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

df[(df['Age']>25) & (df['Transfer_fee']>60000000)]
# lets classify pleyers whether they have high or low . Our threshold is average speed.

#df.info()

df.Transfer_fee.head()

# Basic statistics about the data and its variables

df.describe()
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(df.Transfer_fee)/len(df.Transfer_fee)

df["Transfer_fee"] = ["high" if i > threshold else "low" for i in df.Transfer_fee]

df.loc[:10,["Transfer_fee Level","Transfer_fee"]] # we will learn loc more detailed later
type(df)

df = dataRead()
df.head()  # head shows first 5 rows
# tail shows last 5 rows

df.tail()
# For example lets look frequency of pokemom types

print(df['Position'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example max HP is 255 or min defense is 5

df.describe() #ignore null entries
df.head()
# For example: compare fee value of players that are legendary  or not

# amaç hangi sezonda transfer edilen oyuncuların 

df.boxplot(column='Age',by = 'Season')
# Firstly I create new data from transfer data to explain melt nore easily.



data_new = df.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Transfer_fee','Market_value'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame

data1 = df.head()

data2= df.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = df['Transfer_fee'].head()

data2= df['Market_value'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
df.dtypes
df.info()
# Market value değerlerini kontrol ediyoruz

df["Market_value"].value_counts(dropna =False)

# .Göründüğü gibi 1260 tane NaN var 
df["Market_value"].dropna(inplace = True)
assert 1==1 
assert  df['Market_value'].notnull().all()