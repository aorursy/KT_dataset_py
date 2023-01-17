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
import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset

df=pd.read_csv('../input/pubg-train/train_V2.csv')
df_copy=df.copy() #copy of original dataset
df_copy.sample(10)
#damageDealt can not be in point 



df_copy['damageDealt']=round(df_copy['damageDealt'])
df.shape
df.isnull().sum() #only one null value will drop it
df.dropna(inplace=True)
df.shape
df_copy.info()
# damageDealt will be in int data type

# longestKill will be in int data type

# matchType will be category data type



df_copy['damageDealt']=df_copy['damageDealt'].astype('int')

df_copy['longestKill']=df_copy['longestKill'].astype('int')

df_copy['matchType']=df_copy['matchType'].astype('category')
df_copy.info()
df_copy.columns.values
df_copy['assists'].sort_values(ascending=False) #it is highly unlikely to do zero assist and win the game 
df_copy=df_copy[(df_copy['assists']>0)] #removing them who did zero assists
df_copy.shape
df_copy['boosts'].sort_values(ascending=False) #it is highly unlikely to carry more than 10 boots
df_copy=df_copy[(df_copy['boosts']<=10)] #removing them who carried more than 10 boosts
df_copy['damageDealt'].sort_values(ascending=False) #damageDealt is highly unlikely to be zero 
df_copy=df_copy[(df_copy['damageDealt']>0)]
df_copy['DBNOs'].value_counts() #it is highlt unlike that some group has knocked more than 25 enemies
df_copy=df_copy[df_copy['DBNOs']<=25]
df_copy['headshotKills'].value_counts() #no need to do anything
df_copy.columns.values
df_copy['heals'].value_counts() # it is highly inlikely to carry more than 15 to 20 healings bag will get full and ammo cant be carried
df_copy=df_copy[df_copy['heals']<=20]
df_copy['killPlace'].value_counts() #more kills will give a better rank
df_copy['killPoints'].sort_values(ascending=False) #not required column
df_copy.drop(columns='killPoints',inplace=True)
df_copy.columns.values
df_copy['killStreaks'].value_counts() #nothing needs to e done
df_copy['kills'].value_counts() #it is highly unlikely to kill more than 20 to 25 enemies for a single person
df_copy=df_copy[df_copy['kills']<=25]
df_copy['longestKill'].sort_values(ascending=False) #the minimum distance between can not be zero they can be cheating
df_copy=df_copy[df_copy['longestKill']>0]
df_copy['matchDuration'].sort_values(ascending=False) #no need to be done
df_copy.columns.values
df_copy['matchType'].value_counts() #nothinh needs to be done
df_copy['maxPlace'].value_counts() #col not needed 
df_copy['numGroups'].value_counts() #col not required
df_copy['rankPoints'].value_counts() #col not required
df_copy.drop(columns=['maxPlace','numGroups','rankPoints'],inplace=True)
df_copy.columns.values
df_copy['revives'].value_counts() #it is highly unlikely to revive more than 10 times 
df_copy=df_copy[df_copy['revives']<=10]
df_copy['rideDistance'].sort_values(ascending=False) #nothing needs to be done
df_copy['roadKills'].sort_values(ascending=False) #nothing needs to be done
df_copy['swimDistance'].sort_values(ascending=False) #it is highly unlikely that someone has swimed more than 1000 he may die of the zone
df_copy=df_copy[df_copy['swimDistance']<1000]
df_copy['winPoints'].value_counts() #Nothing needs to be done
df_copy['teamKills'].value_counts()
df_copy[(df_copy['teamKills']==0) & (df_copy['kills']>0)].shape #it is false data ... if kill is more than zero teamkills can not be zero... data is not given properly in this column
df_copy.drop(columns='teamKills',inplace=True)
df_copy.columns.values
df_copy['vehicleDestroys'].value_counts() #can be correct data nothing needs to be done
df_copy['walkDistance'].sort_values(ascending=False) #who has walked zero distance are likely to be cheating ... should walk at least 100m
df_copy=df_copy[df_copy['walkDistance']>100]
df_copy['weaponsAcquired'].value_counts() # it is highly unlikely that someone will change their weapon more than 10 times max
df_copy=df_copy[df_copy['weaponsAcquired']<=10]
df_copy[(df_copy['winPoints']==0) & (df_copy['winPlacePerc']>0)] #it is strange to see that without winpoints how anyone can have any winning chances merely impossible
df_copy.drop(columns='winPoints',inplace=True)
df_copy['winPlacePerc']=df_copy['winPlacePerc']*100
df_copy['winPlacePerc'].value_counts()
df_copy['winPlacePerc'].sort_values(ascending=False) #who has zero winning chance has no use here
df_copy=df_copy[df_copy['winPlacePerc']>0]
df_copy
df.describe()
#for matchType column



plt.figure(figsize=(30,8))

sns.countplot(df_copy['matchType'])

plt.show()
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['matchType'],y=df_copy['winPlacePerc'])
# for assists col



plt.figure(figsize=(30,8))

sns.countplot(df_copy['assists'])

plt.show()
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['assists'],y=df_copy['winPlacePerc'])
# for damageDealt column



plt.figure(figsize=(30,8))

sns.distplot(df_copy['damageDealt'])

plt.show()
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['damageDealt'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.boxplot(df_copy['damageDealt'])

plt.show()
df_copy
winners=df_copy[df_copy['winPlacePerc']>=100] #grops won the matches
winners
plt.figure(figsize=(30,8))

sns.boxplot(winners['damageDealt'])

plt.show()
plt.figure(figsize=(30,8))

sns.distplot(df_copy['walkDistance'])
plt.figure(figsize=(30,8))

sns.distplot(winners['walkDistance'])
plt.figure(figsize=(30,8))

sns.countplot(winners['boosts'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['boosts'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['heals'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['heals'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['matchType'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['matchType'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['assists'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['assists'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['DBNOs'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['DBNOs'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['weaponsAcquired'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['weaponsAcquired'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['headshotKills'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['headshotKills'])
df_copy.corr()
winners.corr()
df_copy.columns.values
plt.figure(figsize=(30,8))

sns.countplot(winners['kills'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['kills'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.countplot(winners['revives'])
plt.figure(figsize=(30,10))

sns.boxplot(x=df_copy['revives'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.scatterplot(x=df_copy['assists'],y=df_copy['winPlacePerc'])
plt.figure(figsize=(30,8))

sns.scatterplot(x=winners['assists'],y=winners['DBNOs'])
df_copy.columns.values
plt.figure(figsize=(30,8))

sns.countplot(winners['walkDistance'])
df_new=df_copy.copy()
df_new #data which can to used for Machine Learning purpose