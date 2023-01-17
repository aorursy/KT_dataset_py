# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
poke = pd.read_csv('../input/pokemon/Pokemon.csv')
poke.head()
poke.info()
# Adapting the columns titles (Changing into upper case)

poke.columns = poke.columns.str.upper().str.replace('_', '')

poke.head()
#Checking the value count of 'TYPE 1' pokemon

poke['TYPE 1'].value_counts()
#Checking the value count of 'TYPE 2' pokemon

poke['TYPE 2'].value_counts()
#Checking null values

poke.isnull().sum()
# Showing Ice types with highest SP.ATK

poke[poke['TYPE 1']=='Ice'].sort_values('SP. ATK', ascending=False).head()
#Set column 'NAME' as index

poke = poke.set_index('NAME')

print(poke)
# Drop the columns # with axis=1

poke = poke.drop(['#'], axis=1)
#Filling null values of type 2 with type 1 as the pokemon might only have one type

poke['TYPE 2'].fillna(poke['TYPE 1'], inplace=True)
#Removing all the text before 'Mega' using regex method

poke.index = poke.index.str.replace('.*(?=Mega)', '')
#return the pokemon with highest ATK

print('Máx ATK: ', poke['ATTACK'].idxmax())
#Calling the describe method on our Data

poke.describe().T
sns.scatterplot(x='HP', y='SPEED', data=poke, hue='LEGENDARY', hue_order=[True, False])
#using subplots with relplot

sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', col ='LEGENDARY', hue_order=[True, False], col_wrap=2)

plt.show()
sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', col ='TYPE 1', col_wrap=4)

plt.show()
sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', col ='LEGENDARY', size='GENERATION') #Subgroups with point size

plt.show()
sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', col ='LEGENDARY', size='GENERATION', hue='GENERATION') #Subgroups with point size and hue

plt.show()
sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', col ='LEGENDARY', size='GENERATION', hue='GENERATION', style='GENERATION') #Subgroups with point style

plt.show()
sns.relplot(x='HP', y='SPEED', data=poke, kind='scatter', alpha=0.4) #Changing point transparency

plt.show()
sns.relplot(x= 'GENERATION', y='ATTACK', data=poke, hue='LEGENDARY', kind='line', style='LEGENDARY', markers=True, ci='sd')
# countplot() vs catplot()

g = sns.countplot(x='TYPE 1', data=poke)

g.set(xlabel ='TYPE 1', ylabel='Quant.')

plt.xticks(rotation=90)

plt.show()
#Bar plots

f, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='TYPE 1', y='SPEED', data=poke)

plt.xticks(rotation=90)

plt.show()
g = sns.catplot(x='LEGENDARY', y='DEFENSE', data=poke, kind='box', order=[True, False]) #we can omit the outliers using 'sym = ""' 

plt.show()