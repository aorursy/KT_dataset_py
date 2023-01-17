# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
inputData = pd.read_csv('../input/train_V2.csv')
inputData.columns
inputData['matchType'].unique()
inputData1 = inputData[~inputData.winPlacePerc.isnull()]
plt.hist(inputData1.winPlacePerc)
corr = inputData.corr()
plt.figure(figsize=(10, 16))
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
corr.winPlacePerc
correlatedcolsWithTarget = list(corr[(corr.winPlacePerc>0.2) | (corr.winPlacePerc<-0.2)]['winPlacePerc'].index)
corr = inputData[correlatedcolsWithTarget].corr()

plt.figure(figsize=(10, 10))
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
# killPlace is highly correlated correlated with damageDealt, kills, killStreaks
# We can keep kills and remove killPlace, damageDealt, killStreaks
correlatedcolsWithTargetRemoveMultiColl = [col for col in correlatedcolsWithTarget if col not in ['killPlace','damageDealt','killStreaks']]
correlatedcolsWithTargetRemoveMultiColl.append('matchType')
correlatedcolsWithTargetRemoveMultiColl
val = inputData[['matchType']].groupby(['matchType']).size()
val.plot.bar()


squad_fpp = inputData[inputData['matchType']=='squad-fpp']
train, test, y_train, y_test = train_test_split(inputData1, inputData1['matchType'], test_size=0.9, random_state=42, stratify=inputData1['matchType'])
cols = ['assists',
 'boosts',
 'DBNOs',
 'headshotKills',
 'heals',
 'kills',
 'longestKill',
 'revives',
 'rideDistance',
 'walkDistance',
 'weaponsAcquired']
i=0
fig = plt.figure(figsize=(20, 50))
plt.subplots_adjust(wspace = 0.5,hspace=2 )
i = 840
for c in cols[1:9]:
    i+=1
#     print(i)
    ax = fig.add_subplot(i)
    val = train[['matchType',c]].groupby(['matchType'])[c].sum()
#     print(val)
    val.plot(kind='bar',ax=ax)
    plt.ylabel(c)
#     plt.show()
# val

ax = fig.add_subplot(i)
val = train[['matchType','walkDistance']].groupby(['matchType'])['walkDistance'].sum()
val.plot(kind='bar',ax=ax)
plt.ylabel('walkDistance')

ax = fig.add_subplot(i)
val = train[['matchType','weaponsAcquired']].groupby(['matchType'])['weaponsAcquired'].sum()
val.plot(kind='bar',ax=ax)
plt.ylabel('weaponsAcquired')

######## Match Type can be neglected as this graph is similar to overall counts graph
matchTypeArr = inputData1.matchType.unique()
fig = plt.figure(figsize=(20, 50))
# plt.subplots_adjust(wspace = 0.5,hspace=2 )
i=1
for matchType in matchTypeArr:
    ax1=plt.subplot(8, 4, i)
    matchTypeDF = inputData1[inputData1['matchType']==matchType]
    plt.hist(matchTypeDF.winPlacePerc)
    plt.xlabel(matchType)
#     plt.show()
    plt.subplot(ax1)
    i+=1
matchTypeArr = inputData1.matchType.unique()
fig = plt.figure(figsize=(20, 50))
# plt.subplots_adjust(wspace = 0.5,hspace=2 )
i=1
for matchType in matchTypeArr:
    ax1=plt.subplot(8, 4, i)
    matchTypeDF = inputData1[inputData1['matchType']==matchType]
    plt.scatter(matchTypeDF.assists, matchTypeDF.winPlacePerc)
    plt.xlabel(matchType)
#     plt.show()
    plt.subplot(ax1)
    i+=1
matchTypeArr = inputData1.matchType.unique()
fig = plt.figure(figsize=(20, 50))
# plt.subplots_adjust(wspace = 0.5,hspace=2 )
i=1
for matchType in matchTypeArr:
    print(i)
    ax1=plt.subplot(8, 4, i)
    matchTypeDF = inputData1[inputData1['matchType']==matchType]
    plt.scatter(matchTypeDF.boosts, matchTypeDF.winPlacePerc)
    plt.xlabel(matchType+" boosts")
    plt.ylabel(matchType+" winPlacePerc")
#     plt.show()
    plt.subplot(ax1)
    i+=1
matchTypeArr = inputData1.matchType.unique()
fig = plt.figure(figsize=(20, 50))
# plt.subplots_adjust(wspace = 0.5,hspace=2 )
i=1
for matchType in matchTypeArr:
    print(i)
    ax1=plt.subplot(8, 4, i)
    matchTypeDF = inputData1[inputData1['matchType']==matchType]
    plt.scatter(matchTypeDF.DBNOs, matchTypeDF.winPlacePerc)
    plt.xlabel(matchType+" DBNOs")
    plt.ylabel(matchType+" winPlacePerc")
#     plt.show()
    plt.subplot(ax1)
    i+=1


train, test, y_train, y_test = train_test_split(inputData1[correlatedcolsWithTargetRemoveMultiColl].drop('winPlacePerc',axis=1), inputData1.winPlacePerc, test_size=0.2, random_state=42, stratify=inputData1['matchType'])
train_matchType = pd.get_dummies(train.matchType, dtype=int).drop('squad-fpp',axis=1)
test_matchType = pd.get_dummies(test.matchType, dtype=int).drop('squad-fpp',axis=1)
x_train = pd.concat([train, train_matchType],axis=1).drop('matchType',axis=1)
x_test = pd.concat([test, test_matchType],axis=1).drop('matchType',axis=1)
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
r2_score(y_test,y_pred)
actualTest = pd.read_csv('../input/test_V2.csv')
actualTest1 = actualTest[[col for col in correlatedcolsWithTargetRemoveMultiColl if col not in ['winPlacePerc']]]
actual_test_matchType = pd.get_dummies(actualTest1.matchType, dtype=int).drop('squad-fpp',axis=1)
x_actual_test = pd.concat([actualTest1, actual_test_matchType],axis=1).drop('matchType',axis=1)
actualTest['winPlacePerc'] = regr.predict(x_actual_test)
actualTest[['Id','winPlacePerc']].to_csv('submission.csv',index=False)


