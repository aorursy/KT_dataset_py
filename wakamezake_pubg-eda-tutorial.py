import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/train_V2.csv")
train.head()
train.shape
print(train["matchType"].unique())
print(len(train["matchType"].unique()))
# どうやら欠損値が含まれている模様
train.isna().sum()
train[train['winPlacePerc'].isna()]
# 欠損値は削除！
train.drop(2744604, inplace=True)
# チーターを見つける
# チーターの情報を学習に使ってしまうと正確な予測ができなくなる
# とりあえず動かずにkillしてるのはチーターとみなす
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
train[train['killsWithoutMoving'] == True].shape
train[train['killsWithoutMoving'] == True]
# 動かずにキルしているプレイヤーは削除
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
# キル多すぎるやつがいるチーターだろ
print(train[train["kills"] > 30].shape)
train[train["kills"] > 30].head()
# キル数多すぎるプレイヤーは削除
train.drop(train[train['kills'] > 30].index, inplace=True)
# ヘッドショットキル / キル
# ヘッドショットレートを計算して、ヘッドショットレート1.0でキル数が多すぎるやつはチーター
train['headshot_rate'] = train['headshotKills'] / train['kills']
# 欠損値のために0で埋めておく
train['headshot_rate'] = train['headshot_rate'].fillna(0)
# と思ったけど現実的にありえそうな範囲だった・・・
print(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)
train[(train['headshot_rate'] == 1) & (train['kills'] > 9)]
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# catb_model = CatBoostRegressor() 
xgb_model = XGBRegressor()
# lgb_model = LGBMRegressor()
# データ多すぎて学習が終わらないのでとりあえず 5万件
sample = train.sample(n=50000)
x = sample.drop(columns=["Id", "groupId", "matchId", "matchType", "winPlacePerc"])
y = sample["winPlacePerc"]
# catb_model.fit(x, y)
xgb_model.fit(x, y)
# lgb_model.fit(x, y)
from xgboost import plot_importance
# どの変数がモデルの精度向上に影響しているかを見てみる
plot_importance(xgb_model)
