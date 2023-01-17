## Importing some libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
df=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

reserved=df.copy()
df.shape
df.head()
df.info()
df.columns
df['rankPoints'].value_counts()
df.drop(columns={'rankPoints'},inplace=True)
df['assists']=df['assists'].astype('int16')

df['boosts']=df['boosts'].astype('int16')

df['DBNOs']=df['DBNOs'].astype('int16')

df['headshotKills']=df['headshotKills'].astype('int16')

df['heals']=df['heals'].astype('int16')

df['killPlace']=df['killPlace'].astype('int16')

df['killPoints']=df['killPoints'].astype('int32')

df['kills']=df['kills'].astype('int16')

df['killStreaks']=df['killStreaks'].astype('int16')

df['matchDuration']=df['matchDuration'].astype('int32')

df['maxPlace']=df['maxPlace'].astype('int16')

df['numGroups']=df['numGroups'].astype('int16')

df['roadKills']=df['roadKills'].astype('int16')

df['teamKills']=df['teamKills'].astype('int16')

df['vehicleDestroys']=df['vehicleDestroys'].astype('int16')

df['weaponsAcquired']=df['weaponsAcquired'].astype('int16')

df['winPoints']=df['winPoints'].astype('int32')

df['winPlacePerc']=df['winPlacePerc'].astype('float32')

df['damageDealt']=df['damageDealt'].astype('float32')

df['longestKill']=df['longestKill'].astype('float32')

df['rideDistance']=df['rideDistance'].astype('float32')

df['swimDistance']=df['swimDistance'].astype('float32')

df['walkDistance']=df['walkDistance'].astype('float32')

df['matchType']=df['matchType'].astype('category')
print(df.shape)

df.info()
plt.figure(figsize=(20,8))

sns.heatmap(df.corr())
sns.jointplot(x='killPoints',y='winPoints',data=df)
sns.jointplot(x='kills',y='damageDealt',data=df,height=8)
sns.jointplot(x='walkDistance',y='winPlacePerc',data=df,height=15)

plt.show()
plt.figure(figsize=(20,6))

sns.jointplot(x='boosts',y='walkDistance',data=df,height=10,color='green')
sns.jointplot(x='winPlacePerc',y='boosts',data=df,height=10,kind='scatter',color='red')
plt.figure(figsize=(20,6))

sns.boxenplot(x='kills',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.boxplot(x='assists',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.boxplot(x='boosts',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.boxenplot(x='DBNOs',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.boxenplot(x='headshotKills',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.boxenplot(x='heals',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.violinplot(x='matchType',y='winPlacePerc',data=df)
plt.figure(figsize=(20,6))

sns.jointplot(x='rideDistance',y='winPlacePerc',data=df,color='green',height=8)
plt.figure(figsize=(20,6))

sns.jointplot(x='swimDistance',y='winPlacePerc',data=df,color='green',height=8)
plt.figure(figsize=(20,10))

sns.countplot(df['matchType'])