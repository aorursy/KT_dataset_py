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
city_info_df = pd.read_csv("/kaggle/input/exam-for-students20200527/city_info.csv")

city_info_df = city_info_df.rename(columns={"Latitude": "CityLatitude", "Longitude": "CityLongitude"})

city_info_df
station_info_df = pd.read_csv("/kaggle/input/exam-for-students20200527/station_info.csv")

station_info_df = station_info_df.rename(columns={"Station": "NearestStation", "Latitude": "StationLatitude", "Longitude": "StationLongitude"})

station_info_df
train_org_df = pd.read_csv("/kaggle/input/exam-for-students20200527/train.csv")

train_org_df
test_org_df = pd.read_csv("/kaggle/input/exam-for-students20200527/test.csv")

test_org_df
train_df = train_org_df.copy()
train_df = pd.merge(train_df, city_info_df, on=["Prefecture", "Municipality"], how="left")

train_df = pd.merge(train_df, station_info_df, on="NearestStation", how="left")
train_df = train_df[~train_df["CityLatitude"].isnull()]
train_df["TradePriceLog"] = np.log(train_df["TradePrice"])

train_df["AreaLog"] = np.log(train_df["Area"])
train_df["MinTimeToNearestStationFlag"] = 1

train_df["MinTimeToNearestStationFlag"] = train_df["MinTimeToNearestStationFlag"].mask(train_df["MinTimeToNearestStation"].isnull(), 0)

train_df["MaxTimeToNearestStationFlag"] = 1

train_df["MaxTimeToNearestStationFlag"] = train_df["MaxTimeToNearestStationFlag"].mask(train_df["MaxTimeToNearestStation"].isnull(), 0)
train_df["MinTimeToNearestStation"] = train_df["MinTimeToNearestStation"].mask(train_df["MinTimeToNearestStation"].isnull(), 120)

train_df["MaxTimeToNearestStation"] = train_df["MaxTimeToNearestStation"].mask(train_df["MaxTimeToNearestStation"].isnull(), 120)
train_df["StationInfoFlag"] = 1

train_df["StationInfoFlag"] = train_df["StationInfoFlag"].mask(train_df["StationLatitude"].isnull(), 0)
train_df["StationLatitude"] = train_df["StationLatitude"].mask(train_df["StationLatitude"].isnull(), train_df["CityLatitude"])

train_df["StationLongitude"] = train_df["StationLongitude"].mask(train_df["StationLongitude"].isnull(), train_df["CityLongitude"])
train_data_columns = [

    "AreaLog",

    "MinTimeToNearestStationFlag",

    "MinTimeToNearestStation",

    "MaxTimeToNearestStation",

    "MaxTimeToNearestStationFlag",

    "CityLatitude",

    "CityLongitude",

    "StationInfoFlag",

    "StationLatitude",

    "StationLongitude",

]

train_data_df = train_df[train_data_columns]



train_tp_df = train_df[["TradePriceLog"]]
import lightgbm as lgbm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

%matplotlib inline
x_train, x_test, y_train, y_test = train_test_split(train_data_df, train_tp_df, test_size=0.2, random_state=42)
lgb_train = lgbm.Dataset(x_train, y_train)

lgb_eval = lgbm.Dataset(x_test, y_test, reference=lgb_train)
lgbm_params = {

    "metric": "rmse",

    "max_depth": 9,

}
gbm = lgbm.train(lgbm_params, lgb_train, valid_sets=lgb_eval, num_boost_round=10000, early_stopping_rounds=100, verbose_eval=50)
test_df = test_org_df.copy()
test_df = pd.merge(test_df, city_info_df, on=["Prefecture", "Municipality"], how="left")

test_df = pd.merge(test_df, station_info_df, on="NearestStation", how="left")
test_df["AreaLog"] = np.log(test_df["Area"])
test_df["MinTimeToNearestStationFlag"] = 1

test_df["MinTimeToNearestStationFlag"] = test_df["MinTimeToNearestStationFlag"].mask(test_df["MinTimeToNearestStation"].isnull(), 0)

test_df["MaxTimeToNearestStationFlag"] = 1

test_df["MaxTimeToNearestStationFlag"] = test_df["MaxTimeToNearestStationFlag"].mask(test_df["MaxTimeToNearestStation"].isnull(), 0)
test_df["MinTimeToNearestStation"] = test_df["MinTimeToNearestStation"].mask(test_df["MinTimeToNearestStation"].isnull(), 120)

test_df["MaxTimeToNearestStation"] = test_df["MaxTimeToNearestStation"].mask(test_df["MaxTimeToNearestStation"].isnull(), 120)
test_df["StationInfoFlag"] = 1

test_df["StationInfoFlag"] = test_df["StationInfoFlag"].mask(test_df["StationLatitude"].isnull(), 0)
test_df["StationLatitude"] = test_df["StationLatitude"].mask(test_df["StationLatitude"].isnull(), test_df["CityLatitude"])

test_df["StationLongitude"] = test_df["StationLongitude"].mask(test_df["StationLongitude"].isnull(), test_df["CityLongitude"])
test_data_columns = [

    "AreaLog",

    "MinTimeToNearestStationFlag",

    "MinTimeToNearestStation",

    "MaxTimeToNearestStation",

    "MaxTimeToNearestStationFlag",

    "CityLatitude",

    "CityLongitude",

    "StationInfoFlag",

    "StationLatitude",

    "StationLongitude",

]

test_data_df = test_df[test_data_columns]
submission_org_df = pd.read_csv("/kaggle/input/exam-for-students20200527/sample_submission.csv")
submission_df = submission_org_df.copy()

# submission_df
submission_df["TradePrice"] = np.exp(gbm.predict(test_data_df))
submission_df
submission_df.to_csv("submission.csv", index=False)