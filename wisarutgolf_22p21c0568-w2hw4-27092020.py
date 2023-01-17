import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
def read_plant_1():

    df1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

    df2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

    return df1,df2
def read_plant_2():

    df1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')

    df2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

    return df1,df2
def create_df(df1,df2):

    df1 = df1.drop('PLANT_ID', axis='columns')

    df2 = df2.drop(['PLANT_ID','SOURCE_KEY'],axis='columns')

    df1["DATE_TIME"] = pd.to_datetime(df1["DATE_TIME"])

    df2["DATE_TIME"] = pd.to_datetime(df2["DATE_TIME"])

    df = pd.merge(df2,df1,on="DATE_TIME",how="inner")

    df['Time'] = df['DATE_TIME'].apply(lambda x : str(x).split(' ')[1])

    df['TOTAL_YIELD_1'] = df.groupby(['SOURCE_KEY','Time']).shift(4)['TOTAL_YIELD'] 

    df['TOTAL_YIELD_2'] = df.groupby(['SOURCE_KEY','Time']).shift(5)['TOTAL_YIELD'] 

    df['TOTAL_YIELD_3'] = df.groupby(['SOURCE_KEY','Time']).shift(6)['TOTAL_YIELD']

    df['TOTAL_YIELD_4'] = df.groupby(['SOURCE_KEY','Time']).shift(7)['TOTAL_YIELD']

    df['TOTAL_YIELD_5'] = df.groupby(['SOURCE_KEY','Time']).shift(8)['TOTAL_YIELD']

    df['TOTAL_YIELD_6'] = df.groupby(['SOURCE_KEY','Time']).shift(9)['TOTAL_YIELD']

    df['TOTAL_YIELD_7'] = df.groupby(['SOURCE_KEY','Time']).shift(10)['TOTAL_YIELD']

    df = df.drop(['DATE_TIME','Time','SOURCE_KEY','DAILY_YIELD','DC_POWER','AC_POWER'],axis =1)

    return df
def make_fold(df):

    data = [df[df.index % 10 == i] for i in range(10)]

    return data
def create_model():

    XGB = XGBRegressor()

    CB = CatBoostRegressor()

    LGB = LGBMRegressor()

    LR = LinearRegression()

    RF = RandomForestRegressor()
def xgb(data):

    train = data[1:]

    test  = data[0]

    train = pd.concat(train)

    train = train.dropna()

    test = test.dropna()

    start = time.time()

    XGB.fit(train.drop(['TOTAL_YIELD'],axis=1),train['TOTAL_YIELD'])

    end = time.time()

    times.append(end-start)

    XGB_pred = XGB.predict(test.drop(['TOTAL_YIELD'],axis=1))

    mae = metrics.mean_absolute_error(XGB_pred, test['TOTAL_YIELD'])

    MAE.append(mae)

    mse = metrics.mean_squared_error(XGB_pred, test['TOTAL_YIELD'])

    MSE.append(mse)

    rmse = np.sqrt(metrics.mean_squared_error(XGB_pred, test['TOTAL_YIELD']))

    RMSE.append(rmse)

    r2 = XGB.score(test.drop(['TOTAL_YIELD'],axis=1), test['TOTAL_YIELD'])

    R2.append(r2)
def cb(data):

    train = data[1:]

    test  = data[0]

    train = pd.concat(train)

    train = train.dropna()

    test = test.dropna()

    start = time.time()

    CB.fit(train.drop(['TOTAL_YIELD'],axis=1),train['TOTAL_YIELD'])

    end = time.time()

    times.append(end-start)

    CB_pred = CB.predict(test.drop(['TOTAL_YIELD'],axis=1))

    mae = metrics.mean_absolute_error(CB_pred, test['TOTAL_YIELD'])

    MAE.append(mae)

    mse = metrics.mean_squared_error(CB_pred, test['TOTAL_YIELD'])

    MSE.append(mse)

    rmse = np.sqrt(metrics.mean_squared_error(CB_pred, test['TOTAL_YIELD']))

    RMSE.append(rmse)

    r2 = CB.score(test.drop(['TOTAL_YIELD'],axis=1), test['TOTAL_YIELD'])

    R2.append(r2)
def lgb(data):

    train = data[1:]

    test  = data[0]

    train = pd.concat(train)

    train = train.dropna()

    test = test.dropna()

    start = time.time()

    LGB.fit(train.drop(['TOTAL_YIELD'],axis=1),train['TOTAL_YIELD'])

    end = time.time()

    times.append(end-start)

    LGB_pred = LGB.predict(test.drop(['TOTAL_YIELD'],axis=1))

    mae = metrics.mean_absolute_error(LGB_pred, test['TOTAL_YIELD'])

    MAE.append(mae)

    mse = metrics.mean_squared_error(LGB_pred, test['TOTAL_YIELD'])

    MSE.append(mse)

    rmse = np.sqrt(metrics.mean_squared_error(LGB_pred, test['TOTAL_YIELD']))

    RMSE.append(rmse)

    r2 = LGB.score(test.drop(['TOTAL_YIELD'],axis=1), test['TOTAL_YIELD'])

    R2.append(r2)
def lr(data):

    train = data[1:]

    test  = data[0]

    train = pd.concat(train)

    train = train.dropna()

    test = test.dropna()

    start = time.time()

    LR.fit(train.drop(['TOTAL_YIELD'],axis=1),train['TOTAL_YIELD'])

    end = time.time()

    times.append(end-start)

    LR_pred = LR.predict(test.drop(['TOTAL_YIELD'],axis=1))

    mae = metrics.mean_absolute_error(LR_pred, test['TOTAL_YIELD'])

    MAE.append(mae)

    mse = metrics.mean_squared_error(LR_pred, test['TOTAL_YIELD'])

    MSE.append(mse)

    rmse = np.sqrt(metrics.mean_squared_error(LR_pred, test['TOTAL_YIELD']))

    RMSE.append(rmse)

    r2 = LR.score(test.drop(['TOTAL_YIELD'],axis=1), test['TOTAL_YIELD'])

    R2.append(r2)
def rf(data):

    train = data[1:]

    test  = data[0]

    train = pd.concat(train)

    train = train.dropna()

    test = test.dropna()

    start = time.time()

    RF.fit(train.drop(['TOTAL_YIELD'],axis=1),train['TOTAL_YIELD'])

    end = time.time()

    times.append(end-start)

    RF_pred = RF.predict(test.drop(['TOTAL_YIELD'],axis=1))

    mae = metrics.mean_absolute_error(RF_pred, test['TOTAL_YIELD'])

    MAE.append(mae)

    mse = metrics.mean_squared_error(RF_pred, test['TOTAL_YIELD'])

    MSE.append(mse)

    rmse = np.sqrt(metrics.mean_squared_error(RF_pred, test['TOTAL_YIELD']))

    RMSE.append(rmse)

    r2 = LR.score(test.drop(['TOTAL_YIELD'],axis=1), test['TOTAL_YIELD'])

    R2.append(r2)
def report(label,MAE,MSE,RMSE,R2,times):

    result = []

    for i in range(len(label)):

        result.append([label[i],MAE[i],MSE[i],RMSE[i],R2[i],times[i]])

    result_df = pd.DataFrame(np.array(result),columns=['Model','Mean Absolute Error','Mean Squared Error','Root Mean Square Error','R-Squared','Time'])

    return result_df
def pipeline(param):

    df = create_df(param[0],param[1])

    data = make_fold(df)

    create_model()

    xgb(data)

    cb(data)

    lgb(data)

    lr(data)

    rf(data)
times = []

MAE = []

MSE = []

RMSE = []

R2 = []

label = ['XG Boost','Cat Boost','Light GBM','Linear Regression','Random Forest']

pipeline(read_plant_1())
report(label,MAE,MSE,RMSE,R2,times)
times = []

MAE = []

MSE = []

RMSE = []

R2 = []

label = ['XG Boost','Cat Boost','Light GBM','Linear Regression','Random Forest']

pipeline(read_plant_2())
report(label,MAE,MSE,RMSE,R2,times)