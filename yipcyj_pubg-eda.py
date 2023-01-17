import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import random
random.seed(88)

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train_V2.csv')
display(df.info())
display(df.head())
display(df.describe())
# There is a single row where winPlacePerc is null.
display(df[df.winPlacePerc.isnull()])
df.dropna(how="all", subset=["winPlacePerc"], inplace=True)
display(df[df.winPlacePerc.isnull()])
df['playersInMatch'] = df.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(15,10))
plt.title('Players in match')
sns.countplot(df[df['playersInMatch'] >= 75]['playersInMatch'])
plt.show()
df[df['playersInMatch'] < 75]['playersInMatch'].describe()
df['playersInTeam'] = df.groupby('groupId')['groupId'].transform('count')

plt.figure(figsize=(15,10))
plt.title('Players in team')
sns.countplot(df[df['playersInTeam'] <= 10]['playersInTeam'])
plt.show()
df[df['playersInTeam'] > 10]['playersInTeam'].describe()
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, fmt=".1f")
plt.show()
normalize_factor = 1 + ((100 - df['playersInMatch']) / 100)
df['killsNorm'] = df.kills * normalize_factor
df['damageDealtNorm'] = df.damageDealt * normalize_factor

cols_to_show = [
    'playersInMatch',
    'kills','killsNorm',
    'damageDealt', 'damageDealtNorm',
]
df[cols_to_show][0:20]
df['healsAndBoosts'] = df['heals'] + df['boosts']
df['totalDistance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
df['headshotRate'] = df['headshotKills'] / df['kills']
df['headshotRate'].fillna(0, inplace=True)
###
# Outliers
###
# Kills without movement
idx = (df['totalDistance'] == 0) & (df['kills'] > 0)
df['anomalyKillsWithoutMovement'] = idx
df[idx].describe()
# Excessive road kills
threshold = df[df['roadKills'] > 0]['roadKills'].quantile(0.999)
print(threshold)
idx = df['roadKills'] > threshold
df['anomalyRoadKills'] = idx
df[idx]
# Excessive kills

# plt.figure(figsize=(15,10))
# sns.countplot(df[df['kills' > df[df['kills'] > 0]['kills'].quantile(0.999)]]['kills'])
# plt.show()

idx = df['kills'] > 30
df['anomalyKills'] = idx
df[idx].describe()
# There is zombie game mode that affects the data.
# Damage dealt is huge. The average in whole dataset is around 200.
# Average number of players in match is 32. It's low.
# Always headshots
idx = (df['headshotRate'] == 1) & (df['kills'] > 9)
df[idx]
# No odd pattern jumps out from these players.
print("99.9% have longestKill less than {}".format(df['longestKill'].quantile(0.999)))

idx = (df['longestKill'] > 1000)
df[idx]
# I don't see anything wrong with these players with at least one kill greater than 1 km.
# Removing outliers
idx = ((df['anomalyKillsWithoutMovement'] == True) | 
    (df['anomalyRoadKills'] == True) |
    (df['anomalyKills'] == True))
print("Dropping {} rows".format(len(df[idx])))
df.drop(df[idx].index, inplace=True)
df.drop(['anomalyKillsWithoutMovement', 'anomalyRoadKills', 'anomalyKills'], axis=1, inplace=True)
# One-hot encode match type
df = pd.get_dummies(df, columns=['matchType'])
def train_test_split(df, test_size=0.1):
    match_ids = df.matchId.unique().tolist()
    train_len = int(len(match_ids) * (1 - test_size))
    train_match_ids = set(random.sample(match_ids, train_len))
    
    train = df[df.matchId.isin(train_match_ids)]
    test = df[~df.matchId.isin(train_match_ids)]
    
    return train, test

df_train, df_val = train_test_split(df)
df = df.dropna()

cols_to_not_fit = [
    'winPlacePerc',
    'Id', 'groupId', 'matchId', 'killPlace', 'killPoints', 'maxPlace', 'rankPoints', 'winPoints',
    'kills', 'damageDealt',
]
cols_to_fit = [ col for col in df.columns.values if col not in cols_to_not_fit ]
cols_to_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(df_train[cols_to_fit], df_train['winPlacePerc'])
val_pred = model.predict(df_val[cols_to_fit])
mean_absolute_error(df_val['winPlacePerc'], val_pred)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=40,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
)
model.fit(df_train[cols_to_fit], df_train['winPlacePerc'])
val_pred = model.predict(df_val[cols_to_fit])
mean_absolute_error(df_val['winPlacePerc'], val_pred)
# Feature importance

from fastai.imports import *
from fastai.structured import *

# feature_importance = rf_feat_importance(model, df_train.drop(columns=['winPlacePerc']))
# feature_importance[:10]
# Kaggle submission

df_test = pd.read_csv("../input/test_V2.csv")
print("length of df_test is {}".format(len(df_test)))

df_test['playersInMatch'] = df_test.groupby('matchId')['matchId'].transform('count')
df_test['playersInTeam'] = df_test.groupby('groupId')['groupId'].transform('count')

normalize_factor = 1 + ((100 - df_test['playersInMatch']) / 100)
df_test['killsNorm'] = df_test.kills * normalize_factor
df_test['damageDealtNorm'] = df_test.damageDealt * normalize_factor

df_test['healsAndBoosts'] = df_test['heals'] + df_test['boosts']
df_test['totalDistance'] = df_test['walkDistance'] + df_test['rideDistance'] + df_test['swimDistance']
df_test['headshotRate'] = df_test['headshotKills'] / df_test['kills']
df_test['headshotRate'].fillna(0, inplace=True)

df_test['anomalyKillsWithoutMovement'] = (df_test['totalDistance'] == 0) & (df_test['kills'] > 0)
df_test['anomalyRoadKills'] = (df_test['roadKills'] > 8)
df_test['anomalyKills'] = (df_test['kills'] > 30)

df_test = pd.get_dummies(df_test, columns=['matchType'])

test_pred = np.clip(a=model.predict(df_test[cols_to_fit]), a_min=0, a_max=1)
print("length of predictions is {}".format(len(test_pred)))
df_submission = pd.DataFrame({
    "Id": df_test["Id"],
    "winPlacePerc": test_pred,
})
df_submission.to_csv("submission.csv", index=False)
display(df_submission.head())
display(df_submission.tail())
