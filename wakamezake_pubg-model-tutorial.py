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
def get_features(df, is_remove=True):
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    if is_remove:
        # とりあえず動かずにkillしてるのはチーターとみなす
        killsWithoutMoving = ((df['kills'] > 0) & (df['totalDistance'] == 0))
        # 動かずにキルしているプレイヤーは削除
        df.drop(df[killsWithoutMoving == True].index, inplace=True)
        # キル多すぎるやつがいるチーターだろ
        TooManykills = train['kills'] > 30
        df.drop(df[TooManykills].index, inplace=True)
    return df
# データ多すぎて学習が終わらないのでとりあえず 5万件
sample = train.sample(n=50000)
sample_copy = sample.copy()
sample = get_features(sample, is_remove=False)
sample_drop = get_features(sample_copy, is_remove=True)
print(sample.shape)
print(sample_drop.shape)
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
xgb_model_1 = XGBRegressor()
xgb_model_2 = XGBRegressor()
lgbm_model_1 = LGBMRegressor()
lgbm_model_2 = LGBMRegressor()
def drop_axis(df):
    x = df.drop(columns=["Id", "groupId", "matchId", "matchType", "winPlacePerc"])
    y = df["winPlacePerc"]
    return x, y
sample_x, sample_y = drop_axis(sample)
sample_drop_x, sample_drop_y = drop_axis(sample_drop)
xgb_model_1.fit(sample_x, sample_y)
lgbm_model_1.fit(sample_x, sample_y)
xgb_model_2.fit(sample_drop_x, sample_drop_y)
lgbm_model_2.fit(sample_drop_x, sample_drop_y)
from xgboost import plot_importance as xgb_plot
from lightgbm import plot_importance as lgbm_plot
# どの変数がモデルの精度向上に影響しているかを見てみる
xgb_plot(xgb_model_1)
xgb_plot(xgb_model_2)
lgbm_plot(lgbm_model_1)
lgbm_plot(lgbm_model_2)
# xgboostのみ
# チーター情報を使わなかった場合
xgb_model_1_predict = xgb_model_1.predict(sample_x)
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_y,
                                                                 xgb_model_1_predict)))
# lightgbmのみ
# チーター情報を使わなかった場合
lgbm_model_1_predict = lgbm_model_1.predict(sample_x)
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_y,
                                                                 lgbm_model_1_predict)))
# stakkingしてみる
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_y,
                                                                 0.5 * xgb_model_1_predict + 0.5 * lgbm_model_1_predict)))
# xgboostのみ
# チーター情報を使った場合
xgb_model_2_predict = xgb_model_2.predict(sample_drop_x)
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_drop_y,
                                                                 xgb_model_2_predict)))
# lightgbmのみ
# チーター情報を使った場合
lgbm_model_2_predict = lgbm_model_2.predict(sample_drop_x)
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_drop_y,
                                                                 lgbm_model_2_predict)))
# stakkingしてみる
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(sample_drop_y,
                                                                 0.5 * xgb_model_2_predict + 0.5 * lgbm_model_2_predict)))
