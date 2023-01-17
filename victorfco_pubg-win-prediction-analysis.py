# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization

from sklearn.model_selection import KFold

from sklearn import svm

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
train.info()
train.head()
train.tail()
print("Total of NaN values on train dataset: ", train.isna().sum().sum())
train.isna().sum()
train[train.isna().any(axis=1)]
# Removing outlier

train.drop(train[train.isna().any(axis=1)].index, inplace=True)
# Creating a new feature of number of players joined in a match

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(25,10))

sns.countplot(train['playersJoined'])

plt.title('Players Joined in a Match')

plt.show()
train['matchType'].unique().tolist()
# Removing event and custom matches

train = train[train.matchType.isin(['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp'])]
group = train.groupby(['matchId','groupId','matchType'])['Id'].count().to_frame('players').reset_index()

group.info()

group.head()

# Non-standard matches

to_remove = group[group['players'] > 4].matchId.unique().tolist()

print(len(to_remove), "matches don't agree with the standard PUBG gameplay.")



# Removing those matchId's from train dataframe

group = group[~group.matchId.isin(to_remove)]
fig, ax = plt.subplots(2, 3, figsize=(16, 8))

for mt, ax in zip(['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp'], ax.ravel()):

    ax.set_xlabel(mt)

    group[group['matchType'] == mt]['players'].value_counts().sort_index().plot.bar(ax=ax)
group.loc[(group['matchType'] == 'solo-fpp') & (group['players'] > 1)].tail()
train[train.groupId == '07b6286649f1e5']
train[train.groupId == '9512eb0b2c0d24']
# Removing matches not agreeing with standard PUBG gameplay

to_remove = group.loc[(group['matchType'] == 'solo') & (group['players'] > 1)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'solo-fpp') & (group['players'] > 1)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'duo') & (group['players'] > 2)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'duo-fpp') & (group['players'] > 2)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]
fig, ax = plt.subplots(2, 3, figsize=(16, 8))

for mt, ax in zip(['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp'], ax.ravel()):

    ax.set_xlabel(mt)

    group[group['matchType'] == mt]['players'].value_counts().sort_index().plot.bar(ax=ax)
# Removing more event and custom matches from the actual train dataset

train = train[train.matchId.isin(group.matchId.unique().tolist())]



print("We have now only {} matches.".format(len(train.matchId.unique().tolist())))
g = sns.PairGrid(train, y_vars=['winPlacePerc'], x_vars=['rankPoints', 'killPoints', 'winPoints'], height=5)

g.map(sns.scatterplot)
g = sns.PairGrid(train, y_vars=['winPlacePerc'], x_vars=['boosts', 'heals'], height=5)

g.map(sns.scatterplot)
g = sns.PairGrid(train, y_vars=['winPlacePerc'], x_vars=['walkDistance', 'rideDistance', 'swimDistance'], height=5)

g.map(sns.scatterplot)
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['boostsHeals'] = train['boosts'] + train['heals']

train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

train['boostsHealsPerTotalDistance'] = train['boostsHeals']/(train['totalDistance']+1) # To avoid infity cases, we add 1 for cases where totalDistance might be 0 and bootsHeals > 0
# One hot encode matchType

train = pd.get_dummies(train, columns=['matchType'])



# Turn groupId and match Id into categorical types

train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')



# Get category coding for groupId and matchID

train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes



# Get rid of old columns

train.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

# train[['groupId_cat', 'matchId_cat']].head()



# Take a look at the encoding

# matchType_encoding = train.filter(regex='matchType')

# matchType_encoding.head()
# Subsampling dataset

#sample = 30000

#train_sample = train.sample(sample)



# Split dataset into training data and target variable



#X = train_sample.drop(columns = ['Id', 'winPlacePerc']) # Subsampled X

#y = train_sample['winPlacePerc'] # Subsampled y



X = train.drop(columns = ['Id', 'winPlacePerc'])

y = train['winPlacePerc']



kf = KFold(n_splits=10)

reg = linear_model.LinearRegression(n_jobs = -1)

outcomes = []

fold = 0



start = time.time()

for train_index, test_index in kf.split(X):

        fold += 1

        X_train, X_test = X.values[train_index], X.values[test_index]

        y_train, y_test = y.values[train_index], y.values[test_index]

        reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        print("Coefficients:", reg.coef_)

        mse = mean_squared_error(y_test, predictions)

        outcomes.append(mse)

        print("Fold {0} MSE: {1}".format(fold, mse))     

mean_outcome = np.mean(outcomes)

print("Average MSE: {0}".format(mean_outcome))



end = time.time()

print("Elapsed time:",(end-start))



# Checking predicted values against the real ones (only last fold)

df = pd.DataFrame().from_dict({'predicted':predictions,'truth':y_test})

df.head()



# Loading test dataset



test = pd.read_csv('../input/test_V2.csv')

print("Total of NaN values on test dataset: ", test.isna().sum().sum())
# Creating playersJoined feature

test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')



# Removing event and custom matches

test = test[test.matchType.isin(['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp'])]

group = test.groupby(['matchId','groupId','matchType'])['Id'].count().to_frame('players').reset_index()



# Non-standard matches

to_remove = group[group['players'] > 4].matchId.unique().tolist()



# Removing those matchId's from train dataframe

group = group[~group.matchId.isin(to_remove)]



# Removing matches not agreeing with standard PUBG gameplay

to_remove = group.loc[(group['matchType'] == 'solo') & (group['players'] > 1)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'solo-fpp') & (group['players'] > 1)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'duo') & (group['players'] > 2)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



to_remove = group.loc[(group['matchType'] == 'duo-fpp') & (group['players'] > 2)].matchId.unique().tolist()

group = group[~group.matchId.isin(to_remove)]



# Removing more event and custom matches from the actual test dataset

test = test[test.matchId.isin(group.matchId.unique().tolist())]



# Creating the same features as train dataset

test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['boostsHeals'] = test['boosts'] + test['heals']

test['totalDistance'] = test['walkDistance'] + test['rideDistance'] + test['swimDistance']

test['boostsHealsPerTotalDistance'] = test['boostsHeals']/(test['totalDistance']+1) # To avoid infity cases, we add 1 for cases where totalDistance might be 0 and bootsHeals > 0



# Working on those non-categorical values

# One hot encode matchType

test = pd.get_dummies(test, columns=['matchType'])



# Turn groupId and match Id into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes



# Get rid of old columns

test.drop(columns=['groupId', 'matchId'], inplace=True)



# Final test dataset for prediction without Id

test_pred = test.loc[:, ~test.columns.isin(['Id'])]
test_pred.head()
predictions = reg.predict(test_pred)

pred_df = pd.DataFrame({'Id' : test['Id'], 'winPlacePerc' : predictions})



# Submission file

pred_df.to_csv("submission.csv", index=False)
pred_df.head()