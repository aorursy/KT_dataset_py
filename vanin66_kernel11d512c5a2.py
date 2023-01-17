import pandas as pd
import numpy as np
train = pd.read_csv("../input/train_V2.csv")
train.head()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']][5:8]
train['healsAndBoosts'] = train['heals']+train['boosts']
train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train['boostsPerWalkDistance'].fillna(0, inplace=True)
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train['healsPerWalkDistance'].fillna(0, inplace=True)
train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1) #The +1 is to avoid infinity.
train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
train[['walkDistance', 'boosts', 'boostsPerWalkDistance' ,'heals',  'healsPerWalkDistance', 'healsAndBoosts', 'healsAndBoostsPerWalkDistance']][40:45]

train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
train['killsPerWalkDistance'].fillna(0, inplace=True)
train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)
train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]
train['headshotPerKill'] = train['headshotKills'] / train['kills']
train['headshotPerKill'].fillna(0, inplace=True) 
train['assistsPerDamage'] = train['assists'] / (train['damageDealt'] + 1)
train['assistsPerDamage'].fillna(0, inplace=True)
# На большее количество примеров выдает ошибку
X = train.drop(['Id', 'killPoints', 'matchId', 'groupId', 'matchType', 'winPlacePerc'], axis = 1)[:300000]
y = train['winPlacePerc'][:300000]
train.shape
from sklearn import ensemble, metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rf = ensemble.RandomForestRegressor(n_estimators = 50, 
                                                max_depth = 10, 
                                                random_state = 1)
-cross_val_score(rf, X, y, cv = 5, scoring='neg_mean_absolute_error').mean()
test = pd.read_csv('../input/test_V2.csv')
test.head()
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')
test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)
test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)
test['healsAndBoosts'] = test['heals']+test['boosts']
test['totalDistance'] = test['walkDistance']+test['rideDistance']+test['swimDistance']
test['boostsPerWalkDistance'] = test['boosts']/(test['walkDistance']+1)
test['boostsPerWalkDistance'].fillna(0, inplace=True)
test['healsPerWalkDistance'] = test['heals']/(test['walkDistance']+1)
test['healsPerWalkDistance'].fillna(0, inplace=True)
test['healsAndBoostsPerWalkDistance'] = test['healsAndBoosts']/(test['walkDistance']+1) 
test['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
test['killsPerWalkDistance'] = test['kills']/(test['walkDistance']+1)
test['killsPerWalkDistance'].fillna(0, inplace=True)
test['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in test['numGroups']]
test['headshotPerKill'] = test['headshotKills'] / test['kills']
test['headshotPerKill'].fillna(0, inplace=True) 
test['assistsPerDamage'] = test['assists'] / (test['damageDealt'] + 1)
test['assistsPerDamage'].fillna(0, inplace=True)
X_test = test.drop(['Id', 'killPoints', 'matchId', 'groupId', 'matchType'], axis = 1)
rf.fit(X, y)
ans = np.clip(rf.predict(X_test), a_min = 0.0, a_max = 1.0)
df_pred = pd.DataFrame({'Id' : test['Id'], 'winPlacePerc' : ans})
df_pred.to_csv("submission.csv", index=False)
