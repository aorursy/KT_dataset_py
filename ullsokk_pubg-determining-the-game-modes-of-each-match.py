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
train = pd.read_csv('../input/train.csv')

matches = train[['matchId']].drop_duplicates()

print(len(matches.index))
#lets look at the number of players, to ensure there are no duplicates (same ID in different matches)
players = train[['Id']].drop_duplicates()

print(len(players.index))
print(len(train.index))


hist = train.hist(column='numGroups',bins=100)
#number of players in each match
matchSize = train.groupby(['matchId']).size().to_frame('matchSize')
train = train.merge(matchSize, how='left', on=['matchId'])

#number of players in each squad
squadSize = train.groupby(['groupId']).size().to_frame('squadSize')
train = train.merge(squadSize, how='left', on=['groupId'])

#Max squad size per match
maxSquadSize= train.groupby('matchId').max()['squadSize'].to_frame('maxSquadSize')
train = train.merge(maxSquadSize, how='left', on=['matchId'])

#Mean squad size per match
meanSquadSize= train.groupby('matchId').mean()['squadSize'].to_frame('meanSquadSize')
train = train.merge(meanSquadSize, how='left', on=['matchId'])

#Median squad size per match
medianSquadSize= train.groupby('matchId').median()['squadSize'].to_frame('medianSquadSize')
train = train.merge(medianSquadSize, how='left', on=['matchId'])

#take a look at som matches with more than 50 groups (which one could argue means a solo game) but that has more than 4 players in one team 
train[['matchId','numGroups','maxSquadSize','meanSquadSize','medianSquadSize']].drop_duplicates().where(train['numGroups'] > 50).where(train['maxSquadSize'] > 4).sort_values(by=['matchId']).head(101)


#make max(squadSize function to determine game type)
def gameType(size):
    if size == 1:
        return "Solo"
    elif size == 2:
        return "Duo"
    elif 2 < size <= 4:
        return "Squads"
    elif size >= 5:
        return "Custom"

train['gameType'] = train['maxSquadSize'].apply(gameType)

matches = train[['matchId','gameType']].drop_duplicates()
matches.groupby('gameType').size()


matches = train[['matchId','numGroups','gameType','maxSquadSize','medianSquadSize','matchSize']].drop_duplicates()
#where doesnt work, use loc in stead
solo = matches.loc[matches['gameType'] == 'Solo']
duo = matches.loc[matches['gameType'] == 'Duo']
squads = matches.loc[matches['gameType'] == 'Squads']
custom = matches.loc[matches['gameType'] == 'Custom']

plt.figure(figsize=(16,15))
plt.subplot(311)

sns.distplot(solo[['numGroups']],label='Solo', color='Skyblue')
sns.distplot(duo[['numGroups']],label='Duo', color='Palegreen')
sns.distplot(squads[['numGroups']],label='Squads', color='Orange')
sns.distplot(custom[['numGroups']],label='Custom games', color='Salmon')
plt.title('number of groups per gametype')
plt.legend()
#plt.show()
plt.subplot(323)
#plt.figure(figsize=(10,10))
plt.title('Solo games')
sns.distplot(solo[['numGroups']],label='Solo', color='Skyblue')
plt.legend()
#plt.show()
plt.subplot(324)
#plt.figure(figsize=(10,10))
plt.title('Duo games')
sns.distplot(duo[['numGroups']],label='Duo', color='Palegreen')
plt.legend()
#plt.show()
plt.subplot(325)
#plt.figure(figsize=(10,10))
plt.title('Squad games')
sns.distplot(squads[['numGroups']],label='Squads', color='Orange')
plt.legend()
#plt.show()
plt.subplot(326)
#plt.figure(figsize=(10,10))
plt.title('Custom games')
sns.distplot(custom[['numGroups']],label='Custom', color='Salmon')
plt.legend()
#plt.show()



plt.figure(figsize=(16,15))
plt.subplot(331)
plt.title('Number of players per match')
sns.distplot(matches[['matchSize']],color='Skyblue')
plt.subplot(332)
few = matches.loc[matches['matchSize'] <= 80]
plt.title('Distribution for less than 80 players')
sns.distplot(few[['matchSize']],color='Skyblue')
plt.subplot(333)
fewer = matches.loc[matches['matchSize'] <= 60]
plt.title('Distribution for less than 60 players')
sns.distplot(fewer[['matchSize']],color='Skyblue')


variable = 'matchSize'

plt.figure(figsize=(16,15))
plt.subplot(311)
sns.distplot(solo[[variable]],label='Solo', color='Skyblue')
sns.distplot(duo[[variable]],label='Duo', color='Palegreen')
sns.distplot(squads[[variable]],label='Squads', color='Orange')
sns.distplot(custom[[variable]],label='Custom games', color='Salmon')
plt.title('number of players per gametype')
plt.legend()
#plt.show()
plt.subplot(323)
#plt.figure(figsize=(10,10))
plt.title('Solo games')
sns.distplot(solo[[variable]],label='Solo', color='Skyblue')
plt.legend()
#plt.show()
plt.subplot(324)
#plt.figure(figsize=(10,10))
plt.title('Duo games')
sns.distplot(duo[[variable]],label='Duo', color='Palegreen')
plt.legend()
#plt.show()
plt.subplot(325)
#plt.figure(figsize=(10,10))
plt.title('Squad games')
sns.distplot(squads[[variable]],label='Squads', color='Orange')
plt.legend()
#plt.show()
plt.subplot(326)
#plt.figure(figsize=(10,10))
plt.title('Custom games')
sns.distplot(custom[[variable]],label='Custom', color='Salmon')
plt.legend()
#plt.show()
