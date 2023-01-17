# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.ensemble import RandomForestRegressor

from datetime import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

import calendar

pd.options.mode.chained_assignment = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
train = pd.read_csv("../input/bike-sharing-demand/train.csv")

test = pd.read_csv("../input/bike-sharing-demand/test.csv")

submit = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv')

train_weather = pd.read_csv("../input/train-weather-last/bike.csv")
print(train_weather.shape)

first_column = train_weather.columns[0]

train_weather_delete = train_weather.drop([first_column],axis=1)

print(train_weather_delete.shape)

train = train_weather_delete
train_weather_delete.tail()
train.tail()
train.info()
train.describe()
test.info()
test.describe()
print ("Shape Of The Before Ouliers: ",train.shape)

train = train[np.abs(train["count"]-train["count"].mean())<=(3*train["count"].std())] 

print ("Shape Of The After Ouliers: ",train.shape)
data = train.append(test, sort=True)

data.shape

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)



data["date"] = data.datetime.apply(lambda x : x.split()[0])

data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])

data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

data.head()
#season = pd.get_dummies(data['season'], drop_first=True)

#data = data.merge(season, left_index=True, right_index=True)

#data.rename(columns={2:'summer',

#                     3:'fall',

#                     4:'winter'},

#                     inplace=True)



#weather = pd.get_dummies(data['weather'], drop_first=True)

#data = data.merge(weather, left_index=True, right_index=True)

#data.rename(columns={2:'mist',

#                     3:'light_rain',

#                     4:'heavy_rain'},

#                     inplace=True)



year = pd.get_dummies(data['year'], drop_first=True)

data = data.merge(year, left_index=True, right_index=True)

data.rename(columns={'2012':'year_2012'},

                     inplace=True)



#weekday = pd.get_dummies(data['weekday'], drop_first=True)

#data = data.merge(weekday, left_index=True, right_index=True)

#data.rename(columns={1:'Monday',

#                     2:'Tuesday',

#                     3:'Wednesday',

#                    4:'Thursday',

#                     5:'Friday',

#                     6:'Saturday'},

#                     inplace=True)





month = pd.get_dummies(data['month'], drop_first=True)

data = data.merge(month, left_index=True, right_index=True)

data.rename(columns={2:'Feb',

                     3:'Mar',

                     4:'Apr',

                     5:'May',

                     6:'Jun',

                     7:'Jul',

                     8:'Aug',

                     9:'Sep',

                     10:'Oct',

                     11:'Nov',

                     12:'Dec'},

                     inplace=True)



hour = pd.get_dummies(data['hour'], drop_first=True)

data = data.merge(hour, left_index=True, right_index=True)



print(data.columns)
data.head()
data.shape
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sns.distplot(data["temp"],ax=axes[0][0])

sns.distplot(data["atemp"],ax=axes[0][1])

sns.distplot(data["humidity"],ax=axes[1][0])

sns.distplot(data["windspeed"],ax=axes[1][1])



axes[0][0].set(xlabel='temp',title="distribution of temp")

axes[0][1].set(xlabel='atemp',title="distribution of atemp")

axes[1][0].set(xlabel='humidity',title="distribution of humidity")

axes[1][1].set(xlabel='windspeed',title="distribution of windspeed")
#dataWind0 = data[data["windspeed"]==0]

#dataWind0
#dataWind0 = data[data["windspeed"]==0]

#dataWindNot0 = data[data["windspeed"]!=0]

#rfModel_wind = RandomForestRegressor(n_estimators=1000,random_state=42)

#windColumns = ["season","weather","humidity","month","temp","year","atemp"]

#rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])



#wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])

#dataWind0.loc[:,"windspeed"] = wind0Values

#data = dataWindNot0.append(dataWind0)

#data.reset_index(inplace=True)

#data.drop('index',inplace=True,axis=1)
sns.distplot(data["windspeed"])
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])

dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]

yLabelsLog = np.log(yLabels)
print(dataTrain.shape)

print(dataTest.shape)
data.head()
# dropFeatures = ['casual',"count","datetime","date","registered"]

dropFeatures = ['casual',"count","datetime","date","registered","season","weather","year","weekday","month","hour","temp","holiday"]

# dropFeatures = ['casual',"count","datetime","date","registered","season","weather","year","weekday","month"]

dataTrain  = dataTrain.drop(dropFeatures,axis=1)

dataTest  = dataTest.drop(dropFeatures,axis=1)
## dropFeatures = ['casual',"count","datetime","date","registered"]

#dropFeatures = ['casual',"count","datetime","date","registered","season","weather","year","weekday","month","hour","temp", "holiday"]

## dropFeatures = ['casual',"count","datetime","date","registered","season","weather","year","weekday","month"]

#dataTrain  = dataTrain.drop(dropFeatures,axis=1)

#dataTest  = dataTest.drop(dropFeatures,axis=1)
sns.distplot(yLabels)
sns.distplot(yLabelsLog)
## random forest

#rfModel = RandomForestRegressor(n_estimators=1000,random_state=42)

#yLabelsLog = np.log(yLabels)

#rfModel.fit(dataTrain,yLabelsLog)

#preds = rfModel.predict(X= dataTrain)
## multiple linear regression

lrmodel = LinearRegression()

lrmodel.fit(dataTrain, yLabelsLog)

r_sq = lrmodel.score(dataTrain, yLabelsLog)

print('coefficient of determination (r_square) = ', r_sq)

print('intercept = ', lrmodel.intercept_)

print('slope = ', lrmodel.coef_)

#y_preds = lrmodel.predict(X= dataTrain)

#y_preds[0:10]
dataTrain_ols = sm.add_constant(dataTrain)

olsmodel = sm.OLS(yLabelsLog, dataTrain_ols).fit()

print(olsmodel.summary())
##polynomial regression

transformer = PolynomialFeatures(degree=2, include_bias=False)

dataTrain_poly = transformer.fit_transform(dataTrain)

prmodel = LinearRegression().fit(dataTrain_poly, yLabelsLog)

r_sq = prmodel.score(dataTrain_poly, yLabelsLog)

print('coefficient of determination (r_square) = ', r_sq)

print('intercept = ', prmodel.intercept_)

print('slope = ', prmodel.coef_)

#y_poly_preds = prmodel.predict(X= train_poly)

#y_poly_preds[0:10]
#predsTest = rfModel.predict(X= dataTest)

#submission = pd.DataFrame({

#        "datetime": datetimecol,

#        "count": [max(0, x) for x in np.exp(predsTest)]

#    })

##submission.to_csv('bike_predictions_RF.csv', index=False)

#submission.to_csv('bike_predictions_RF.csv', index=False)
predsTest = lrmodel.predict(X= dataTest)

submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

##submission.to_csv('bike_predictions_LR.csv', index=False)

submission.to_csv('bike_predictions_LR.csv', index=False)
dataTest_poly = transformer.fit_transform(dataTest)

predsTest_poly = prmodel.predict(X= dataTest_poly)

submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest_poly)]

    })

##submission.to_csv('bike_predictions_PR.csv', index=False)

submission.to_csv('bike_predictions_PR.csv', index=False)