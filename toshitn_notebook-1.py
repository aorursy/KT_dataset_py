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
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.model_selection import KFold

from IPython.core.debugger import Pdb

import lightgbm as lgb

from sklearn.metrics import mean_squared_error
train = pd.read_csv("/kaggle/input/exam-for-students20200527/train.csv")

station_info = pd.read_csv("/kaggle/input/exam-for-students20200527/station_info.csv")

city_info = pd.read_csv("/kaggle/input/exam-for-students20200527/city_info.csv")

test = pd.read_csv("/kaggle/input/exam-for-students20200527/test.csv")

sample_submission = pd.read_csv("/kaggle/input/exam-for-students20200527/sample_submission.csv")
train.head()
station_info.head()
city_info.head()
city_info[city_info["Municipality"]=="Abiko"]
print(train.shape)

print(test.shape)

print(station_info.shape)

print(city_info.shape)
## 都市情報の緯度経度情報のみを使用する

# テーブルの結合

train1 = pd.merge(train, city_info, on='Municipality', how='left')

test1 = pd.merge(test, city_info, on='Municipality', how='left')
# テーブルの結合が想定通りか確認

# train

print(train1.shape)

print(train1["Latitude"].isnull().sum())

print(train1["Longitude"].isnull().sum())



# test

print(test1.shape)

print(test1["Latitude"].isnull().sum())

print(test1["Longitude"].isnull().sum())
#不要列の削除

train1 = train1.drop(["Prefecture_y"], axis=1)

test1 = test1.drop(["Prefecture_y"], axis=1)
## 2点間の距離、方位角を算出する関数

from math import *



# 楕円体

ELLIPSOID_GRS80 = 1 # GRS80

ELLIPSOID_WGS84 = 2 # WGS84



# 楕円体ごとの長軸半径と扁平率

GEODETIC_DATUM = {

    ELLIPSOID_GRS80: [

        6378137.0,         # [GRS80]長軸半径

        1 / 298.257222101, # [GRS80]扁平率

    ],

    ELLIPSOID_WGS84: [

        6378137.0,         # [WGS84]長軸半径

        1 / 298.257223563, # [WGS84]扁平率

    ],

}



# 反復計算の上限回数

ITERATION_LIMIT = 1000



'''

Vincenty法(逆解法)

2地点の座標(緯度経度)から、距離と方位角を計算する

:param lat1: 始点の緯度

:param lon1: 始点の経度

:param lat2: 終点の緯度

:param lon2: 終点の経度

:param ellipsoid: 楕円体

:return: 距離と方位角

'''

# def vincenty_inverse(lat1, lon1, lat2, lon2, ellipsoid=None):

def vincenty_inverse_sibuya(X):    

    

    ellipsoid=None

    

    lat1=X["Latitude"]

    lon1=X["Longitude"]

    

    # 今回は、とりあえずsibuyaで固定

    lat2= 35.6938253

    lon2= 139.7033559



    # 差異が無ければ0.0を返す

    if isclose(lat1, lat2) and isclose(lon1, lon2):

        

        return 0.0



    # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する

    # 楕円体が未指定の場合はGRS80の値を用いる

    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))

    b = (1 - ƒ) * a



    φ1 = radians(lat1)

    φ2 = radians(lat2)

    λ1 = radians(lon1)

    λ2 = radians(lon2)



    # 更成緯度(補助球上の緯度)

    U1 = atan((1 - ƒ) * tan(φ1))

    U2 = atan((1 - ƒ) * tan(φ2))



    sinU1 = sin(U1)

    sinU2 = sin(U2)

    cosU1 = cos(U1)

    cosU2 = cos(U2)



    # 2点間の経度差

    L = λ2 - λ1



    # λをLで初期化

    λ = L



    # 以下の計算をλが収束するまで反復する

    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける

    for i in range(ITERATION_LIMIT):

        sinλ = sin(λ)

        cosλ = cos(λ)

        sinσ = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)

        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ

        σ = atan2(sinσ, cosσ)

        sinα = cosU1 * cosU2 * sinλ / sinσ

        cos2α = 1 - sinα ** 2

        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α

        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))

        λʹ = λ

        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))



        # 偏差が.000000000001以下ならbreak

        if abs(λ - λʹ) <= 1e-12:

            break

    else:

        # 計算が収束しなかった場合はNoneを返す

        return None



    # λが所望の精度まで収束したら以下の計算を行う

    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)

    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))

    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))



    # 2点間の楕円体上の距離

    s = b * A * (σ - Δσ)



    # 各点における方位角

    α1 = atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)

    α2 = atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + pi



    if α1 < 0:

        α1 = α1 + pi * 2



        

    return round(s, 3)







## 時間がなかったのでべた書き・・



# def vincenty_inverse(lat1, lon1, lat2, lon2, ellipsoid=None):

def vincenty_inverse_funabashi(X):    

    

    ellipsoid=None

    

    lat1=X["Latitude"]

    lon1=X["Longitude"]

    

    # 地理的なことを考え船橋を追加

    lat2= 35.6945485

    lon2= 139.9827277



    # 差異が無ければ0.0を返す

    if isclose(lat1, lat2) and isclose(lon1, lon2):

        

        return 0.0



    # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する

    # 楕円体が未指定の場合はGRS80の値を用いる

    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))

    b = (1 - ƒ) * a



    φ1 = radians(lat1)

    φ2 = radians(lat2)

    λ1 = radians(lon1)

    λ2 = radians(lon2)



    # 更成緯度(補助球上の緯度)

    U1 = atan((1 - ƒ) * tan(φ1))

    U2 = atan((1 - ƒ) * tan(φ2))



    sinU1 = sin(U1)

    sinU2 = sin(U2)

    cosU1 = cos(U1)

    cosU2 = cos(U2)



    # 2点間の経度差

    L = λ2 - λ1



    # λをLで初期化

    λ = L



    # 以下の計算をλが収束するまで反復する

    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける

    for i in range(ITERATION_LIMIT):

        sinλ = sin(λ)

        cosλ = cos(λ)

        sinσ = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)

        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ

        σ = atan2(sinσ, cosσ)

        sinα = cosU1 * cosU2 * sinλ / sinσ

        cos2α = 1 - sinα ** 2

        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α

        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))

        λʹ = λ

        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))



        # 偏差が.000000000001以下ならbreak

        if abs(λ - λʹ) <= 1e-12:

            break

    else:

        # 計算が収束しなかった場合はNoneを返す

        return None



    # λが所望の精度まで収束したら以下の計算を行う

    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)

    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))

    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))



    # 2点間の楕円体上の距離

    s = b * A * (σ - Δσ)



    # 各点における方位角

    α1 = atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)

    α2 = atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + pi



    if α1 < 0:

        α1 = α1 + pi * 2



        

    return round(s, 3)
# 渋谷からの距離を特徴量として追加する

train1["distance1"] = train1.apply(vincenty_inverse_sibuya, axis=1)

test1["distance1"] = test1.apply(vincenty_inverse_sibuya, axis=1)



train1["distance2"] = train1.apply(vincenty_inverse_funabashi, axis=1)

test1["distance2"] = test1.apply(vincenty_inverse_funabashi, axis=1)
train1["distance1"].hist()
test1["distance1"].hist()
# 緯度経度情報は削除する

train1 = train1.drop(["Latitude","Longitude"], axis=1)

test1 = test1.drop(["Latitude","Longitude"], axis=1)
train1.isnull().sum()
test1.isnull().sum()
# 欠損が多い列は除外

# カテゴリカル変数は、中央値で埋める

# 数値型データは、平均値で埋める

def fill_na(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    total_cnt = df.shape[0]

    for col in df.columns:

        # 目的変数は処理対象外

        if col == "TradePrice":

            next

        

        col_type = df[col].dtypes

        na_cnt = df[col].isnull().sum()

        

        if  na_cnt / total_cnt > 0.3:

            df = df.drop(columns=col)

        else:        

            if df[col].isnull().sum() > 0:

                if col_type in numerics:

                    df[col] = df[col].fillna(df[col].mean())

                else:

                    #カテゴリは"nan"という文字列で埋めるのもありと思う

                    df[col] = df[col].fillna(str(df[col].mode()))

    

    return df
# trainとtestを纏めて欠損処理を行う

train1["kubun"] = "train"

test1["kubun"] = "test"

test1["TradePrice"] = np.NaN

all_data = pd.concat([train1, test1], axis=0)
# 欠損がなくなっていることを確認

all_data = fill_na(all_data)

all_data.isnull().sum()
# 木系の分類アルゴリズムを採用するので、カテゴリカル変数はすべてLabelEncoderでエンコーディング

le = LabelEncoder()

le_count = 0



for col in all_data:

    if all_data[col].dtype == 'object' and col != "TradePrice" and col != "kubun":

        print(col)

        le.fit(all_data[col])

        all_data[col] = le.transform(all_data[col])



        le_count += 1

            

print('%d columns were label encoded.' % le_count)
# 評価指標がRMSLEなのでターゲットをログ変換してからモデリングする

all_data["TradePrice"] = np.log(all_data["TradePrice"] + 1)
# 教師データとテストデータに再分離する

train2 = all_data[all_data["kubun"] == "train"]

test2 = all_data[all_data["kubun"] == "test"]



train2 = train2.drop(["kubun"], axis=1)

test2 = test2.drop(["kubun", "TradePrice"], axis=1)
# 今回の評価指標

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:

    rmsle = mean_squared_error(np.log1p(y_true), np.log1p(y_pred))

    return np.sqrt(rmsle)
# 不要列の削除

train2 = train2.drop(["id"], axis=1)
y = train2["TradePrice"].values

X = train2.drop("TradePrice", axis=1).values

kf = KFold(n_splits=3, shuffle=True)



first_fold = True

rmsle_ave = 0

epoch = 0

for train_index, test_index in kf.split(X, y):

    

    X_train = X[train_index]

    y_train = y[train_index]

    X_test = X[test_index]

    y_test = y[test_index]    

    

    train_data_set = lgb.Dataset(X_train, y_train)

    test_data_set = lgb.Dataset(X_test, y_test, reference=train_data_set)



    # グリッドサーチは時間かかりそうなので、取り合えず固定・・・

    params = {                                                                                               

        'boosting_type': 'gbdt',                                                                             

        'objective': 'regression_l2',                                                                           

        'metric': 'rmse',                                                                             

        'num_leaves': 40,                                                                                    

        'learning_rate': 0.05,                                                                               

        'feature_fraction': 0.9,                                                                             

        'bagging_fraction': 0.8,                                                                             

        'bagging_freq': 5,   

        'lambda_l2': 2,

    }                                                                                                        



    gbm = lgb.train(params,                                                                                  

                    train_data_set,                                                                               

                    num_boost_round=200,                                                                      

                    valid_sets=test_data_set,                                                                     

                    early_stopping_rounds=10

                    )

    

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 元に戻す

    y_pred = np.exp(y_pred) - 1

    rmsle_val = rmsle(y_test, y_pred)

    rmsle_ave = rmsle_ave + rmsle_val

    epoch = epoch + 1



print('rmsle: {}'.format(rmsle_ave/epoch))
test2.shape
# テストデータの予測

test3 = test2.drop(["id"], axis=1)



y_pred_test = gbm.predict(test3, num_iteration=gbm.best_iteration)

y_pred_test = np.exp(y_pred_test) - 1
sample_submission["TradePrice"] = y_pred_test

sample_submission.to_csv('submission.csv', index=False)