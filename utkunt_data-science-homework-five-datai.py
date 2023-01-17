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
# loading data

data = pd.read_csv('../input/fifa19/data.csv')

data = data.set_index('ID')

data.head()
data['Age'][20801] #indexing by using square brackets
data.Nationality[193080] #indexing by using dot notaion
data.loc[192985,['Club']] #indexing by using loc accessor
data[['Name','Nationality']].head()
#series and dataframes, series got one square brackets, dataframes got two

print(type(data['Nationality']))

print(type(data[['Nationality']]))
#slicing dataframe

data.loc[158023:192985,'Composure':'SlidingTackle']
data.loc[158023:192985,'Marking':]
#filtering by boolean series

boolean = data.Potential > 93

data[boolean]
# multi-filters

filter01 = data.Potential > 90

filter02 = data.Overall > 90

data[filter01 & filter02]
#filtering column-based

data.Name[data.Overall>93]
# transforming data by using 'apply' function

def my_funct(n):

    return n*10

data.GKReflexes.apply(my_funct)
# transforming data by lambda function

data.GKReflexes.apply(lambda n: n*10)
# define new column from existing columns

data['GKOverall'] = data.GKDiving + data.GKHandling + data.GKKicking + data.GKPositioning + data.GKReflexes

data['GKOverall']
# data's index name

print(data.index.name)
# changing data's index name

data.index.name = 'NewID'

print(data.index.name)

data.head()
# redifine index

data.index = range(1,18208,1)

data.index.name = 'Index'

data
# we can change data index by data.index = data['#']
# setting multi-index

newdata = data.set_index(['Nationality','Age'])

newdata.head(100)
# pivot data

new_dic = {'city':['Antalya', 'Mugla', 'Istanbul'],'population_level':['A','C','A'],'rank':[50,70,10]}

df = pd.DataFrame(new_dic)

df.index.name = 'ID'



df.pivot(index='population_level',columns='city',values='rank')
df01 = df.set_index(['population_level','city'])

df01
df01.unstack(level=0)
# change inner and outer indexes

df02 = df01.swaplevel(0,1)

df02
# getting mean values by nation

data.groupby('Nationality').mean()
data.groupby('Club').Age.max()
data.groupby('Nationality')[['Potential','Overall']].mean()