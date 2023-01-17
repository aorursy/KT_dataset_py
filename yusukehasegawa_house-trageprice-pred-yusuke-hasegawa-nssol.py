# 開始時間

from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

print(datetime.now(JST))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 基本

import datetime as dt

import random

import glob

import cv2

import os

from os import path

import gc

import sys

import json

import itertools

import re



# データ加工

import pandas as pd

from pandas import DataFrame, Series

import numpy as np

import scipy as sp

import scipy.sparse as sps



# データ可視化

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib import ticker

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns

from pylab import rcParams



# 検定

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm import tqdm_notebook as tqdm

# 評価

from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, log_loss



# 前処理

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.preprocessing import QuantileTransformer   # 数値列のRankGauss処理用



# モデリング

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier, CatBoostRegressor, FeaturesData, Pool

from lightgbm import LGBMClassifier, LGBMRegressor

from xgboost import XGBClassifier, XGBRegressor



import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau



# DataRobot SDK

#import datarobot as dr

#from datarobot.enums import AUTOPILOT_MODE

#from datarobot.enums import DIFFERENCING_METHOD

#from datarobot.enums import TIME_UNITS



# 表示量

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 100)



# 乱数シード固定

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(72)
# 検定スキームを定義

kf = KFold(n_splits=5, shuffle=True)
# 範囲を最大・最小までにCapする

def capping(series, min_threshold, max_threshold):

    series_filtered = series.copy()

    index_outlier_up = [series_filtered  >= max_threshold]

    index_outlier_low = [series_filtered <= min_threshold]

    series_filtered.iloc[index_outlier_up] = max_threshold

    series_filtered.iloc[index_outlier_low] = min_threshold

    return series_filtered
# check histgram

def check_histgram(X_train, X_test, col):

    plt.figure(figsize=[7,7])

    X_train[col].hist(density=True, alpha=0.5, bins=20, color='b')

    X_test[col].hist(density=True, alpha=0.5, bins=20, color='r')

    plt.xlabel(col)

    plt.ylabel('density')

    plt.show()
# Ordinal Encoding

def ordinal_encode(train_X, test_X, cols):

    oe = OrdinalEncoder(cols=cols)

    oe.fit(train_X[cols])

    train_X[cols] = oe.transform(train_X[cols])

    test_X[cols] = oe.transform(test_X[cols])

    

    return train_X, test_X



# Label Encoding

def label_encode(train_X, test_X, cols):

    le = LabelEncoder()

    le.fit(train_X[cols])

    train_X[cols] = le.transform(train_X[cols])

    test_X[cols] = le.transform(test_X[cols])

    

    return train_X, test_X



# One-hot Encoding

def one_hot_encode(train_X, test_X, cols):

    train_X = pd.get_dummies(train_X, columns=cols)

    test_X = pd.get_dummies(test_X, columns=cols)

    

    return train_X, test_X



# Target Encoding

def target_encode(train_X, test_X, col, train_Y, target):

    # col列と目的変数だけのDataFrameを作成

    temp_X = pd.DataFrame({col: train_X[col], target: train_Y})

    

    # ======= test に対する Target Encoding ================

    # train全体でカテゴリ値ごとの目的変数の平均値を集計

    target_mean = temp_X.groupby(col)[target].mean()

    # testの各カテゴリ値を平均値で置換

    test_X[col] = test_X[col].map(target_mean)

    # nanになってしまった行は「trainのtargetの平均値」で埋める

    test_X[col].fillna(temp_X[target].mean(), inplace=True)

    

    # ======= train に対する Target Encoding ================

    # train の変換後の値を格納するSeriesを準備

    enc_train = Series(np.zeros(len(train_X)), index=train_X.index)

    enc_train[:] = np.nan

    

    # KFoldクラスを用いて、顧客ID単位で分割する

    for i, (train_ix, valid_ix) in enumerate(kf.split(train_X)):

        train_X_ = temp_X.iloc[train_ix]

        valid_X_ = temp_X.iloc[valid_ix]



        # out-of-foldでカテゴリ値ごとの目的変数の平均値を集計

        summary = train_X_.groupby([col])[target].mean()

        enc_train.iloc[valid_ix] = valid_X_[col].map(summary)

        # nanになってしまった行は「train_X_のtargetの平均値」で埋める

        enc_train.iloc[valid_ix].fillna(train_X_[target].mean(), inplace=True)

    

    # trainの各カテゴリ値を平均値で置換

    train_X[col] = enc_train

    

    return train_X, test_X





def log_scaler(train_X, test_X, col):

    train_X[col] = train_X[col].apply(np.log1p)

    test_X[col] = test_X[col].apply(np.log1p)



    return train_X, test_X



def standard_scaler(train_X, test_X, col):

    scaler = StandardScaler()

    train_X[col] = scaler.fit_transform(train_X.iloc[:, train_X.columns == col])

    test_X[col] = scaler.transform(test_X.iloc[:, test_X.columns == col])

    

    return train_X, test_X

# DataFrameを圧縮してメモリ削減



# コピペで使える。Kaggleでの実験を効率化する小技まとめ - 天色グラフィティ

#   https://amalog.hateblo.jp/entry/kaggle-snippets#DataFrame%E3%81%AE%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84%E3%81%99%E3%82%8B



# pandasのDataFrameはintならnp.int64に、floatならnp.float64がデフォルトで使われます。 

# しかし、ある程度データセットが大きくなってくると、DataFrameがメモリを圧迫して学習を思うように進めることができなかったりします。

# そこで、各列の値の範囲を参照し、適切な型に変換します。



import logging



def reduce_mem_usage(df, logger=None, level=logging.DEBUG):

    print_ = print if logger is None else lambda msg: logger.log(level, msg)

    start_mem = df.memory_usage().sum() / 1024**2

    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype

        if col_type != 'object' and col_type != 'datetime64[ns]':

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
# ターゲットの列名

TARGET = 'TradePrice'



# データ読み込み

df_train = pd.read_csv('../input/exam-for-students20200527/train.csv', index_col=0)

df_test = pd.read_csv('../input/exam-for-students20200527/test.csv', index_col=0)

df_submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv')

df_city_info = pd.read_csv('../input/exam-for-students20200527/city_info.csv')

df_station_info = pd.read_csv('../input/exam-for-students20200527/station_info.csv')

'''

df_train = pd.read_csv('../input/exam-for-students20200527/train.csv', index_col=0, skiprows=lambda x: x%90!=0)

df_test = pd.read_csv('../input/exam-for-students20200527/test.csv', index_col=0, skiprows=lambda x: x%90!=0)

df_submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv', skiprows=lambda x: x%90!=0)

df_city_info = pd.read_csv('../input/exam-for-students20200527/city_info.csv')

df_station_info = pd.read_csv('../input/exam-for-students20200527/station_info.csv')

'''



display(df_train)
df_train.dtypes
# 各DataFrameの列名を修正

df_city_info.rename(columns={'Latitude': 'Municipality-Latitude', 'Longitude': 'Municipality-Longitude'}, inplace=True)

df_station_info.rename(columns={'Latitude': 'Station-Latitude', 'Longitude': 'Station-Longitude'}, inplace=True)



# 結合

df_train = pd.merge(left=df_train, right=df_city_info, how='left', on=['Prefecture', 'Municipality'])

df_train = pd.merge(left=df_train, right=df_station_info, how='left', left_on='NearestStation', right_on='Station')

df_test = pd.merge(left=df_test, right=df_city_info, how='left', on=['Prefecture', 'Municipality'])

df_test = pd.merge(left=df_test, right=df_station_info, how='left', left_on='NearestStation', right_on='Station')



# Latitude と Longitude の列を作成

#    station_info でマッチする駅名が無かった場合は city_info から Latitude と Longitude を持ってくる

df_train['Latitude'] = df_train['Station-Latitude']

df_train['Latitude'].mask(df_train['Latitude'].isnull(), df_train['Municipality-Latitude'], inplace=True)

df_train['Longitude'] = df_train['Station-Longitude']

df_train['Longitude'].mask(df_train['Longitude'].isnull(), df_train['Municipality-Longitude'], inplace=True)

df_test['Latitude'] = df_test['Station-Latitude']

df_test['Latitude'].mask(df_test['Latitude'].isnull(), df_test['Municipality-Latitude'], inplace=True)

df_test['Longitude'] = df_test['Station-Longitude']

df_test['Longitude'].mask(df_test['Longitude'].isnull(), df_test['Municipality-Longitude'], inplace=True)





df_train
# [GPS]pyprojを使用してGPSの緯度経度から距離、方位角、仰角を算出する - Qiita

#   https://qiita.com/tomo001/items/84d49e18e9dd3373ce0f



import math

from pyproj import Geod



# ellpsは赤道半径。GPSはWGS84を使っている。距離は6,378,137m

g = Geod(ellps='WGS84')



# Station	Latitude	Longitude

#  Tokyo	35.6812362	139.7671248

df_train['tokyo_Latitude'] = 35.6812362

df_train['tokyo_Longitude'] = 139.7671248

df_test['tokyo_Latitude'] = 35.6812362

df_test['tokyo_Longitude'] = 139.7671248



# 距離を計算

df_train['distance_from_Tokyo']  = g.inv(

    df_train['tokyo_Longitude'].values.tolist(), 

    df_train['tokyo_Latitude'].values.tolist(), 

    df_train['Longitude'].values.tolist(), 

    df_train['Latitude'].values.tolist(), 

)[2]



df_test['distance_from_Tokyo']  = g.inv(

    df_test['tokyo_Longitude'].values.tolist(), 

    df_test['tokyo_Latitude'].values.tolist(), 

    df_test['Longitude'].values.tolist(), 

    df_test['Latitude'].values.tolist(), 

)[2]



display(df_train[['tokyo_Latitude', 'tokyo_Longitude', 'Latitude', 'Longitude', 'distance_from_Tokyo']].head())





# Google Mapを見て、だいたいの埼玉県の中心の緯度・経度を見つける

# https://www.google.co.jp/maps/place/%E5%9F%BC%E7%8E%89%E7%9C%8C/@36.052726,139.1099863,10z/

df_train['center_of_Saitama_Latitude'] = 36.052726

df_train['center_of_Saitama_Longitude'] = 139.387949



# 距離を計算

df_train['distance_from_center_of_Saitama']  = g.inv(

    df_train['center_of_Saitama_Longitude'].values.tolist(), 

    df_train['center_of_Saitama_Latitude'].values.tolist(), 

    df_train['Longitude'].values.tolist(), 

    df_train['Latitude'].values.tolist(), 

)[2]



distance = 75   # 単位:km

print(df_train.shape)

df_train = df_train[df_train['distance_from_center_of_Saitama'] < distance*1000]

print(df_train.shape)



# xとyに分割

train_Y = df_train[TARGET]

train_X = df_train.drop([TARGET], axis=1)

test_X = df_test



# 特徴量エンジニアリング前の、元のカラム名一覧を取っておく（欠損フラグを立てるところで使う）

base_columns = train_X.columns
# 不要な列を削除

train_X = train_X.drop(['Prefecture'], axis=1)

train_X = train_X.drop(['Municipality'], axis=1)

train_X = train_X.drop(['DistrictName'], axis=1)

train_X = train_X.drop(['NearestStation'], axis=1)

train_X = train_X.drop(['TimeToNearestStation'], axis=1)

train_X = train_X.drop(['Station'], axis=1)

train_X = train_X.drop(['tokyo_Latitude'], axis=1)

train_X = train_X.drop(['tokyo_Longitude'], axis=1)

train_X = train_X.drop(['Station-Longitude'], axis=1)

train_X = train_X.drop(['Station-Latitude'], axis=1)

train_X = train_X.drop(['Municipality-Longitude'], axis=1)

train_X = train_X.drop(['Municipality-Latitude'], axis=1)

train_X = train_X.drop(['Longitude'], axis=1)

train_X = train_X.drop(['Latitude'], axis=1)

train_X = train_X.drop(['center_of_Saitama_Longitude'], axis=1)

train_X = train_X.drop(['center_of_Saitama_Latitude'], axis=1)

train_X = train_X.drop(['distance_from_center_of_Saitama'], axis=1)

test_X = test_X.drop(['Prefecture'], axis=1)

test_X = test_X.drop(['Municipality'], axis=1)

test_X = test_X.drop(['DistrictName'], axis=1)

test_X = test_X.drop(['NearestStation'], axis=1)

test_X = test_X.drop(['TimeToNearestStation'], axis=1)

test_X = test_X.drop(['Station'], axis=1)

test_X = test_X.drop(['tokyo_Latitude'], axis=1)

test_X = test_X.drop(['tokyo_Longitude'], axis=1)

test_X = test_X.drop(['Station-Longitude'], axis=1)

test_X = test_X.drop(['Station-Latitude'], axis=1)

test_X = test_X.drop(['Municipality-Longitude'], axis=1)

test_X = test_X.drop(['Municipality-Latitude'], axis=1)

test_X = test_X.drop(['Longitude'], axis=1)

test_X = test_X.drop(['Latitude'], axis=1)



col = 'Type'

print(train_X[col].isnull().sum())   # 欠損数のチェック



# 土地を含むかどうか

f_hasLand = lambda x: 1 if ('Land' in x) else 0

train_X['hasLand'] = train_X[col].apply(f_hasLand)

test_X['hasLand'] = test_X[col].apply(f_hasLand)



# 建物を含むかどうか

f_hasBuilding = lambda x: 1 if ('Residential' in x) or ('Condominium' in x) else 0

train_X['hasBuilding'] = train_X[col].apply(f_hasBuilding)

test_X['hasBuilding'] = test_X[col].apply(f_hasBuilding)



display(train_X[['Type', 'hasLand', 'hasBuilding']])



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'Region'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'MinTimeToNearestStation'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(999, inplace=True)

test_X[col].fillna(999, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'MaxTimeToNearestStation'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(999, inplace=True)

test_X[col].fillna(999, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'FloorPlan'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



#display(train_X[col].unique())



# 部屋数

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)

f_countRooms = lambda x: float(re.search('([1234567890])', str(x)).group()) if re.search('([1234567890])', str(x)) is not None else 0

train_X['Rooms'] = train_X[col].apply(f_countRooms)

test_X['Rooms'] = test_X[col].apply(f_countRooms)



display(train_X[[col, 'Rooms']])



# 分解しつくしたので元の列は削除

train_X = train_X.drop([col], axis=1)

test_X = test_X.drop([col], axis=1)





#train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, 'Rooms')

col = 'Area'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(0, inplace=True)

#test_X[col].fillna(0, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'AreaIsGreaterFlag'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(0, inplace=True)

#test_X[col].fillna(0, inplace=True)



#train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'LandShape'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'Frontage'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'FrontageIsGreaterFlag'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(-999, inplace=True)

#test_X[col].fillna(-999, inplace=True)



train_X[col] = train_X[col].isnull().replace({True : 1, False : 0})

test_X[col] = test_X[col].isnull().replace({True : 1, False : 0})

check_histgram(train_X, test_X, col)

col = 'TotalFloorArea'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'TotalFloorAreaIsGreaterFlag'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(-999, inplace=True)

#test_X[col].fillna(-999, inplace=True)



train_X[col] = train_X[col].isnull().replace({True : 1, False : 0})

test_X[col] = test_X[col].isnull().replace({True : 1, False : 0})

check_histgram(train_X, test_X, col)

col = 'BuildingYear'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(1900, inplace=True)

test_X[col].fillna(1900, inplace=True)



check_histgram(train_X, test_X, col)

col = 'PrewarBuilding'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(1900, inplace=True)

#test_X[col].fillna(1900, inplace=True)



check_histgram(train_X, test_X, col)

col = 'Structure'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



display(train_X[col].unique())



# SRC =鉄骨鉄筋コンクリート を含むかどうか

f_SRC = lambda x: 1 if ('SRC' in x) else 0

train_X['Structure_SRC'] = train_X[col].apply(f_SRC)

test_X['Structure_SRC'] = test_X[col].apply(f_SRC)



# RC =鉄筋コンクリート を含むかどうか

f_RC = lambda x: 1 if ('RC' in x) else 0

train_X['Structure_RC'] = train_X[col].apply(f_RC)

test_X['Structure_RC'] = test_X[col].apply(f_RC)



# S =鉄骨 を含むかどうか

f_S = lambda x: 1 if ('S' in x) else 0

train_X['Structure_S'] = train_X[col].apply(f_S)

test_X['Structure_S'] = test_X[col].apply(f_S)



# LS =軽量鉄骨構造 を含むかどうか

f_LS = lambda x: 1 if ('LS' in x) else 0

train_X['Structure_LS'] = train_X[col].apply(f_LS)

test_X['Structure_LS'] = test_X[col].apply(f_LS)



# B =コンクリートブロック を含むかどうか

f_B = lambda x: 1 if ('B' in x) else 0

train_X['Structure_B'] = train_X[col].apply(f_B)

test_X['Structure_B'] = test_X[col].apply(f_B)



# W =木製 を含むかどうか

f_W = lambda x: 1 if ('W' in x) else 0

train_X['Structure_W'] = train_X[col].apply(f_W)

test_X['Structure_W'] = test_X[col].apply(f_W)



display(train_X[[col, 'Structure_SRC', 'Structure_RC', 'Structure_S', 'Structure_LS', 'Structure_B', 'Structure_W']])



# 分解しつくしたので元の列は削除

train_X = train_X.drop([col], axis=1)

test_X = test_X.drop([col], axis=1)

col = 'Use'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



display(train_X[col].unique())



# House を含むかどうか

f_House = lambda x: 1 if ('House' in x) else 0

train_X[col + 'House'] = train_X[col].apply(f_House)

test_X[col + 'House'] = test_X[col].apply(f_House)



# Complex を含むかどうか

f_Complex = lambda x: 1 if ('Complex' in x) else 0

train_X[col + 'Complex'] = train_X[col].apply(f_Complex)

test_X[col + 'Complex'] = test_X[col].apply(f_Complex)



# Parking Lot を含むかどうか

f_ParkingLot = lambda x: 1 if ('Parking Lot' in x) else 0

train_X[col + 'Parking Lot'] = train_X[col].apply(f_ParkingLot)

test_X[col + 'Parking Lot'] = test_X[col].apply(f_ParkingLot)



# Other を含むかどうか

f_Other = lambda x: 1 if ('Other' in x) else 0

train_X[col + 'Other'] = train_X[col].apply(f_Other)

test_X[col + 'Other'] = test_X[col].apply(f_Other)



# Shop を含むかどうか

f_Shop = lambda x: 1 if ('Shop' in x) else 0

train_X[col + 'Shop'] = train_X[col].apply(f_Shop)

test_X[col + 'Shop'] = test_X[col].apply(f_Shop)



# Office を含むかどうか

f_Office = lambda x: 1 if ('Office' in x) else 0

train_X[col + 'Office'] = train_X[col].apply(f_Office)

test_X[col + 'Office'] = test_X[col].apply(f_Office)



# Workshop を含むかどうか

f_Workshop = lambda x: 1 if ('Workshop' in x) else 0

train_X[col + 'Workshop'] = train_X[col].apply(f_Workshop)

test_X[col + 'Workshop'] = test_X[col].apply(f_Workshop)



# Warehouse を含むかどうか

f_Warehouse = lambda x: 1 if ('Warehouse' in x) else 0

train_X[col + 'Warehouse'] = train_X[col].apply(f_Warehouse)

test_X[col + 'Warehouse'] = test_X[col].apply(f_Warehouse)



# Factory を含むかどうか

f_Factory = lambda x: 1 if ('Factory' in x) else 0

train_X[col + 'Factory'] = train_X[col].apply(f_Factory)

test_X[col + 'Factory'] = test_X[col].apply(f_Factory)



display(train_X[[col, col + 'House', col + 'Complex', col + 'Parking Lot', col + 'Other', col + 'Shop', col + 'Office', col + 'Workshop', col + 'Warehouse', col + 'Factory']])



# 分解しつくしたので元の列は削除

train_X = train_X.drop([col], axis=1)

test_X = test_X.drop([col], axis=1)

col = 'Purpose'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'Direction'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'Classification'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'Breadth'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'CityPlanning'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'CoverageRatio'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



check_histgram(train_X, test_X, col)

col = 'FloorAreaRatio'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(0, inplace=True)

test_X[col].fillna(0, inplace=True)



check_histgram(train_X, test_X, col)

col = 'Year'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(0, inplace=True)

#test_X[col].fillna(0, inplace=True)



check_histgram(train_X, test_X, col)

col = 'Quarter'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

#train_X[col].fillna(0, inplace=True)

#test_X[col].fillna(0, inplace=True)



check_histgram(train_X, test_X, col)

col = 'Renovation'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('Not yet', inplace=True)

test_X[col].fillna('Not yet', inplace=True)



display(train_X[col].unique())



# Done を含むかどうか

f_Done = lambda x: 1 if ('Done' in x) else 0

train_X[col] = train_X[col].apply(f_Done)

test_X[col] = test_X[col].apply(f_Done)





check_histgram(train_X, test_X, col)

col = 'Remarks'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna('NN', inplace=True)

test_X[col].fillna('NN', inplace=True)



train_X, test_X = target_encode(train_X, test_X, col, train_Y, TARGET)

#train_X, test_X = ordinal_encode(train_X, test_X, col)

check_histgram(train_X, test_X, col)

col = 'distance_from_Tokyo'

print(train_X[col].isnull().sum())   # 欠損数のチェック

# 欠損フラグの作成

#train_X[col + '_isna'] = train_X[col].isnull().replace({True : 1, False : 0})

#test_X[col + '_isna'] = test_X[col].isnull().replace({True : 1, False : 0})

train_X[col].fillna(999999, inplace=True)

test_X[col].fillna(999999, inplace=True)



train_X, test_X = log_scaler(train_X, test_X, col)

check_histgram(train_X, test_X, col)

missing = train_X.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

display(missing)



missing = test_X.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

display(missing)
# 欠損値補完

train_X = train_X.replace([np.inf, -np.inf], np.nan)

test_X = test_X.replace([np.inf, -np.inf], np.nan)

train_X.fillna(-9999, inplace=True)

test_X.fillna(-9999, inplace=True)

# すべての列を正規化（for NN）

for col in train_X.columns.values.tolist():

    train_X, test_X = standard_scaler(train_X, test_X, col)

    
train_X
# スコアを計算

def calculate_score(valid_Y_, pred_y_):

    #score = mean_absolute_error(valid_Y_, pred_y_)                 # MAE   Mean Absolute Error                    平均絶対誤差

    #score = np.mean(np.abs((pred_y_ - valid_Y_) / true)) * 100     # MAPE  Mean Absolute Persentage Error         平均絶対誤差率

    #score = mean_squared_error(valid_Y_, pred_y_)                  # MSE   Mean Squared Error                     平均二乗誤差

    #score = np.sqrt(mean_squared_error(valid_Y_, pred_y_))         # RMSE  Root Mean Squared Error                平均平方二乗誤差

    score = np.sqrt(mean_squared_log_error(valid_Y_, pred_y_))     # RMSLE Root Mean Squared Logarithmic Error    対数平方平均二乗誤差

    #score = r2_score(valid_Y_, pred_y_)                            # R2    R-squared                              決定係数

    

    return score



# Neural Networkのパラメータ

kernel_initializer = 'he_normal'

activation = 'relu'

dropout_pct = 0.1

patience = 2

epoch = 30

batch_size = 32

optimizer = 'adam'



# MLP Neural Network モデルを生成

def create_mlp_nn(mlp_inputDim):

    

    # create MLP model

    model = Sequential()

    model.add(Dense(units=512, input_dim=mlp_inputDim, 

                    kernel_initializer=kernel_initializer,activation=activation))    

    model.add(Dropout(dropout_pct))

    model.add(Dense(units=256, kernel_initializer=kernel_initializer, activation=activation))     

    model.add(Dropout(dropout_pct))

    model.add(Dense(units=32, kernel_initializer=kernel_initializer, activation=activation))     

    model.add(Dropout(dropout_pct))

    model.add(Dense(units=1, activation='linear'))     

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error']) 

    

    return model

%%time



# CVごとの各モデルのスコアを保存するDataFrame

df_scores = pd.DataFrame()



# CVごとのFeature Importanceを保存するDataFrame

df_feature_importance = pd.DataFrame()



for i, (train_ix, valid_ix) in enumerate(kf.split(train_X, train_Y)):

    # Random Seed初期化

    seed_everything(72)

    

    # 予測結果を保存するDataFrame

    df_predictions     = pd.DataFrame()

    df_predictions_tmp = pd.DataFrame()

    

    print("============ starting CV#" + str(i) + "==========================================================")



#    train_X_, train_Y_ = train_X.iloc[train_ix].values, train_Y.iloc[train_ix].values

#    valid_X_, valid_Y_ = train_X.iloc[valid_ix].values, train_Y.iloc[valid_ix].values

    train_X_, train_Y_ = train_X.iloc[train_ix], train_Y.iloc[train_ix]

    valid_X_, valid_Y_ = train_X.iloc[valid_ix], train_Y.iloc[valid_ix]

    

    # 学習

    # gb : Gradient Boost------------------------------------------------------

    model_gb = GradientBoostingRegressor()

    # RMSLEで最適化する場合

    model_gb.fit(train_X_, np.log1p(train_Y_))

    pred_y_ = np.expm1(model_gb.predict(valid_X_))

    df_predictions['gb'] = pred_y_

    df_scores.at['gb', 'CV#' + str(i)] = calculate_score(valid_Y_, pred_y_)



    # GradientBoost の Feature Importanceを保存

    feat_imp = pd.Series(model_gb.feature_importances_, index=train_X.columns, name="CV#" + str(i))

    df_feature_importance =  pd.concat([df_feature_importance, feat_imp.to_frame()], axis=1)



    print('lgb : light GBM #------------------------------------------------------')

    model_lgb = LGBMRegressor(boosting_type='gbdt', class_weight='balanced')

    model_lgb.fit(train_X_, np.log1p(train_Y_), eval_metric='rmse')

    pred_y_ = np.expm1(model_lgb.predict(valid_X_))

    df_predictions['lgb'] = pred_y_

    df_scores.at['lgb', 'CV#' + str(i)] = calculate_score(valid_Y_, pred_y_)

    

    print('xgb : XGBoost #------------------------------------------------------')

    model_xgb = XGBRegressor()

    model_xgb.fit(train_X_, np.log1p(train_Y_))

    pred_y_ = np.expm1(model_xgb.predict(valid_X_))

    df_predictions['xgb'] = pred_y_

    df_scores.at['xgb', 'CV#' + str(i)] = calculate_score(valid_Y_, pred_y_)

    

    print('rf : Random Forest #------------------------------------------------------')

    model_rf = RandomForestRegressor()

    model_rf.fit(train_X_, np.log1p(train_Y_))

    pred_y_ = np.expm1(model_rf.predict(valid_X_))

    df_predictions['rf'] = pred_y_

    df_scores.at['rf', 'CV#' + str(i)] = calculate_score(valid_Y_, pred_y_)

    

    print('cb : CatBoost #------------------------------------------------------')

    model_cb = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=5, loss_function='RMSE')

    model_cb.fit(train_X_, np.log1p(train_Y_), early_stopping_rounds=10, logging_level='Silent')

    pred_y_ = np.expm1(model_cb.predict(valid_X_))

    df_predictions['cb'] = pred_y_

    df_scores.at['cb', 'CV#' + str(i)] = calculate_score(valid_Y_, pred_y_)

    '''

    print('nn : Neural Network #------------------------------------------------------')

    scores_nn = []

    df_predictions_nn = pd.DataFrame()

    # Seed Average する

    for seed in range(3):

        print("- - - starting seed#" + str(seed) + "- - - - - - - -")

        seed_everything(seed)



        # callback parameter

        filepath = "nn_best_model_Seed"+str(seed)+".hdf5" 

        es = EarlyStopping(patience=patience, mode='min', verbose=1) 

        checkpoint = ModelCheckpoint(monitor='loss', filepath=filepath, save_best_only=True, mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='loss',  patience=patience, verbose=0,  mode='min')



        # 訓練実行

        mlp_inputDim = train_X_.shape[1]

        model_nn = create_mlp_nn(mlp_inputDim)

        model_nn.fit(

            train_X_, np.log1p(train_Y_),

            epochs=epoch, 

            batch_size=batch_size,

            callbacks=[es, checkpoint, reduce_lr_loss],

            verbose=0

        )



        # load best model weights

        if os.path.exists(filepath):

            model_nn.load_weights(filepath)



        # 予測

        pred_y_ = list(itertools.chain.from_iterable(np.expm1(model_nn.predict(valid_X_, batch_size=batch_size)).reshape((-1,1))))

        df_predictions_nn['nn_Seed#' + str(seed)] = pred_y_

        

        # スコア計算

        scores_nn.append(calculate_score(valid_Y_, pred_y_))

        

    df_predictions['nn'] = df_predictions_nn.mean(axis=1).values

    df_scores.at['nn', 'CV#' + str(i)] = np.mean(scores_nn)

    '''

    print('Average Ensemble - 2 Models #------------------------------------------------------')

    models = list(df_predictions.columns.values)

    for m1, m2 in list(itertools.combinations(models, 2)):

        df_predictions_tmp[m1+'-'+m2] = df_predictions[[m1, m2]].mean(axis=1)

        df_scores.at[m1+'-'+m2, 'CV#' + str(i)] = calculate_score(valid_Y_, df_predictions_tmp[m1+'-'+m2])



    print('Average Ensemble - 3 Models #------------------------------------------------------')

    models = list(df_predictions.columns.values)

    for m1, m2, m3 in list(itertools.combinations(models, 3)):

        df_predictions_tmp[m1+'-'+m2+'-'+m3] = df_predictions[[m1, m2, m3]].mean(axis=1)

        df_scores.at[m1+'-'+m2+'-'+m3, 'CV#' + str(i)] = calculate_score(valid_Y_, df_predictions_tmp[m1+'-'+m2+'-'+m3])



    print('Average Ensemble - 4 Models #------------------------------------------------------')

    models = list(df_predictions.columns.values)

    for m1, m2, m3, m4 in list(itertools.combinations(models, 4)):

        df_predictions_tmp[m1+'-'+m2+'-'+m3+'-'+m4] = df_predictions[[m1, m2, m3, m4]].mean(axis=1)

        df_scores.at[m1+'-'+m2+'-'+m3+'-'+m4, 'CV#' + str(i)] = calculate_score(valid_Y_, df_predictions_tmp[m1+'-'+m2+'-'+m3+'-'+m4])



    print('Average Ensemble - 5 Models #------------------------------------------------------')

    models = list(df_predictions.columns.values)

    for m1, m2, m3, m4, m5 in list(itertools.combinations(models, 5)):

        df_predictions_tmp[m1+'-'+m2+'-'+m3+'-'+m4+'-'+m5] = df_predictions[[m1, m2, m3, m4, m5]].mean(axis=1)

        df_scores.at[m1+'-'+m2+'-'+m3+'-'+m4+'-'+m5, 'CV#' + str(i)] = calculate_score(valid_Y_, df_predictions_tmp[m1+'-'+m2+'-'+m3+'-'+m4+'-'+m5])



        

# シングルモデル間の相関を表示

mask = np.zeros_like(df_predictions.corr(method ='spearman'))

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(5, 7))

    ax = sns.heatmap(df_predictions.corr(method ='spearman'), mask=mask, vmax=1, vmin=0.8, square=True,linewidths=.5,xticklabels=1, yticklabels=1)

plt.show()

    

# CVスコアを表示

df_scores['algo_mean'] = df_scores.mean(axis=1)

display(df_scores.sort_values('algo_mean', ascending=True))

# feature importanceを計算しておく

df_feature_importance['min'] = df_feature_importance.min(axis=1)

df_feature_importance['median'] = df_feature_importance.median(axis=1)

df_feature_importance['max'] = df_feature_importance.max(axis=1)

df_feature_importance['mean'] = df_feature_importance.mean(axis=1)

df_feature_importance['std'] = df_feature_importance.std(axis=1)



# Feature Importanceを描画

sort_by = 'mean'

#sort_by = 'median'



# プロットするデータ

df_feature_importance = df_feature_importance.sort_values(by=[sort_by, 'max'], ascending=True)

x = df_feature_importance.index

y = df_feature_importance[sort_by]

xerr = (df_feature_importance["median"]-df_feature_importance["min"], df_feature_importance["max"]-df_feature_importance["median"])



# サイズ

plt.rcParams['figure.figsize'] = 10,0.25*len(df_feature_importance.index)

plt.rcParams['lines.linewidth'] = 2



# ラベル

plt.title('Feature Importance')

plt.xlabel('Importance')

plt.ylabel('Features')



# プロット

width = 0.8

plt.barh(x, y, width, align='center', xerr=xerr, ecolor='b')



# Gridの設定

plt.xticks(np.arange(min(y), max(y), 0.05))

plt.grid(b=True, which='major', axis='x', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', axis='x', color='#999999', linestyle='-', alpha=0.2)

#plt.yticks([])



# フォント設定

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

plt.rcParams['font.size'] = 10



# グラフ表示とファイル保存

plt.show()

plt.savefig('feature_importance.png',bbox_inches="tight")
# 有用性上位を出力

df_feature_importance = df_feature_importance.sort_values(by=[sort_by, 'max'], ascending=False)

feature_list = df_feature_importance[df_feature_importance[sort_by] > 0.001].index



print(len(feature_list))

print(feature_list.to_list())

%%time

df_predictions = pd.DataFrame()



# GradientBoost

model_gb = GradientBoostingRegressor()

model_gb.fit(train_X, np.log1p(train_Y))

df_predictions['gb'] = np.expm1(model_gb.predict(test_X))



# Light GBM

model_lgb = LGBMRegressor(boosting_type='gbdt', class_weight='balanced')

model_lgb.fit(train_X, np.log1p(train_Y), eval_metric='rmse')

pred_y_ = np.expm1(model_lgb.predict(test_X))

df_predictions['lgb'] = pred_y_



# XGBoost

model_xgb = XGBRegressor()

model_xgb.fit(train_X, np.log1p(train_Y))

pred_y_ = np.expm1(model_xgb.predict(test_X))

df_predictions['xgb'] = pred_y_



# Random Forest

model_rf = RandomForestRegressor()

model_rf.fit(train_X, np.log1p(train_Y))

pred_y_ = np.expm1(model_rf.predict(test_X))

df_predictions['rf'] = pred_y_



# CatBoost

model_cb = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=5, loss_function='RMSE')

model_cb.fit(train_X, np.log1p(train_Y), early_stopping_rounds=10, logging_level='Silent')

pred_y_ = np.expm1(model_cb.predict(test_X))

df_predictions['cb'] = pred_y_



# Average Ensemble + submission

df_submission[TARGET] = df_predictions.mean(axis=1).values

df_submission.to_csv('submission.csv', index=False)

df_submission
