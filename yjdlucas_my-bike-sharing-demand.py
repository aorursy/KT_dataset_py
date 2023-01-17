# coding:utf-8

import numpy as np

import pandas as pd

import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
DataTrain=pd.read_csv(r'../input/train.csv')

DataTest=pd.read_csv(r'../input/test.csv')

DataTrain.head()

DataTest.head()

DataTrain.info()

DataTest.info()

DataTrain.isnull().sum()

DataTest.isnull().sum()
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

fig.set_size_inches(5, 5)

sns.boxplot(data=DataTrain, y="count", orient="v", ax=ax)
DataTrain.reset_index(inplace=True)

fig1,(ax1, ax2) = plt.subplots(nrows=2, ncols=1)

fig1.set_size_inches(10, 5)

sns.regplot(x="index", y="temp", data=DataTrain, ax=ax1)

sns.regplot(x="index", y="atemp", data=DataTrain, ax=ax2)
fig2 = plt.figure()

fig2.set_size_inches(10, 5)

ax3 = fig2.add_subplot(1, 1, 1)

sns.regplot(x="index", y="humidity", data=DataTrain, ax=ax3)
fig3 = plt.figure()

fig3.set_size_inches(10, 5)

ax4 = fig3.add_subplot(1, 1, 1)

sns.regplot(x="index", y="windspeed", data=DataTrain, ax=ax4)
DataTrain['datetime'] = pd.to_datetime(DataTrain['datetime'])

DataTrain['year'] = DataTrain['datetime'].dt.year

DataTrain['month'] = DataTrain['datetime'].dt.month

DataTrain['day'] = DataTrain['datetime'].dt.day

DataTrain['hour'] = DataTrain['datetime'].dt.hour



DataTest['datetime'] = pd.to_datetime(DataTest['datetime'])

DataTest['year'] = DataTest['datetime'].dt.year

DataTest['month'] = DataTest['datetime'].dt.month

DataTest['day'] = DataTest['datetime'].dt.day

DataTest['hour'] = DataTest['datetime'].dt.hour
dataWind0 = DataTrain[DataTrain["windspeed"] == 0]

dataWindNot0 = DataTrain[DataTrain["windspeed"] != 0]

rfModel_wind = RandomForestRegressor()

windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])



wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])

dataWind0["windspeed"] = wind0Values

DataTrain = dataWindNot0.append(dataWind0)

DataTrain.sort_values("datetime", inplace=True)



dataWind1 = DataTest[DataTest["windspeed"] == 0]

dataWindNot1 = DataTest[DataTest["windspeed"] != 0]

rfModel_wind1 = RandomForestRegressor()

rfModel_wind1.fit(dataWindNot1[windColumns], dataWindNot1["windspeed"])



wind1Values = rfModel_wind1.predict(X=dataWind1[windColumns])

dataWind1["windspeed"] = wind1Values

DataTest = dataWindNot1.append(dataWind1)

DataTest.sort_values("datetime", inplace=True)
DataTrain = DataTrain[np.abs(DataTrain["count"]-DataTrain["count"].mean()) <= (3*DataTrain["count"].std())]
fig4, (ax6,ax7,ax8,ax9) = plt.subplots(ncols=4)

fig4.set_size_inches(16, 8)

# season

sns.barplot(x="season", y="count", data=DataTrain, ax=ax6)

# holiday

sns.barplot(x="holiday", y="count", data=DataTrain, ax=ax7)

# working day

sns.barplot(x="workingday", y="count", data=DataTrain, ax=ax8)

# weather

sns.barplot(x="weather", y="count", data=DataTrain, ax=ax9)
fig5, (ax10,ax11,ax12,ax13) = plt.subplots(ncols=4)

fig5.set_size_inches(16, 8)

# year

sns.barplot(x="year", y="count", data=DataTrain, ax=ax10)

# month

sns.barplot(x="month", y="count", data=DataTrain, ax=ax11)

# day

sns.barplot(x="day", y="count", data=DataTrain, ax=ax12)

# hour

sns.barplot(x="hour", y="count", data=DataTrain, ax=ax13)
fig6, (ax14,ax15,ax16,ax17) = plt.subplots(ncols=4)

fig6.set_size_inches(20, 6)

# temp

sns.barplot(x="temp",  y="count", data=DataTrain, ax=ax14)

# atemp

sns.barplot(x="atemp",  y="count", data=DataTrain, ax=ax15)

# humidity

sns.barplot(x="humidity",  y="count", data=DataTrain, ax=ax16)

# windspeed

sns.barplot(x="windspeed",  y="count", data=DataTrain, ax=ax17)
# 相关性分析

Correlation = DataTrain[:].corr()

# Correlation = DataTrain[["season", "holiday", "workingday", "weather", "temp",

#                         "atemp", "humidity", "windspeed", "year", "month", "day", "hour", "count"]].corr()

mask = np.array(Correlation)

mask[np.tril_indices_from(mask)] = False

fig7 = plt.figure()

fig7.set_size_inches(10, 10)

ax5 = fig7.add_subplot(1, 1, 1)

sns.heatmap(Correlation, mask=mask, square=True, annot=True, cbar=True, ax=ax5)
# 补充分析

fig8,(ax18,ax19) = plt.subplots(nrows=2)

holiday_hour = pd.DataFrame(DataTrain.groupby(["hour","holiday"],sort=True)["count"].mean()).reset_index()

sns.pointplot(x=holiday_hour["hour"], y=holiday_hour["count"],hue=holiday_hour["holiday"], data=holiday_hour, join=True,ax=ax18)

ax18.set(xlabel='Hour', ylabel='count',title="Count group by Hour Of holiday", label='big')



workingday_hour = pd.DataFrame(DataTrain.groupby(["hour","workingday"],sort=True)["count"].mean()).reset_index()

sns.pointplot(x=workingday_hour["hour"], y=workingday_hour["count"],hue=workingday_hour["workingday"], data=workingday_hour, join=True,ax=ax19)

ax19.set(xlabel='Hour', ylabel='count',title="Count group by Hour Of workingday",label='big')
# 确定特征变量

FeatureNames = ["season","holiday","workingday","weather","temp","humidity","windspeed","month","year","hour"]

yLabels = DataTrain["count"]

DataTrain = DataTrain[FeatureNames]

datetimecol = DataTest["datetime"]

DataTest = DataTest[FeatureNames]
def rmsle(y, y_, convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
# 建立模型

# Initialize logistic regression model

lModel = LinearRegression()

# Train the model

yLabelsLog = np.log1p(yLabels)

lModel.fit(X=DataTrain, y=yLabelsLog)

# Make predictions

preds = lModel.predict(X=DataTrain)

print("RMSLE Value For Linear Regression: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))
predsTest = lModel.predict(X=DataTest)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission.to_csv(r'bike_prediction.csv', index=False)