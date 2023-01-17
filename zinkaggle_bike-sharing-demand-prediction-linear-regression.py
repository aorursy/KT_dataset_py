# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import data set

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

train_data.head(3)
#check Null values

train_data.isnull().values.any()
#Root Mean Squared Logarithmic Error

def rmsle(prediction, actual):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in prediction]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
#construct linear regression model 1

#model 1 includes all features

train_x = train_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)

train_y = train_data['count'].values

lrm_1 = linear_model.LinearRegression().fit(train_x, train_y)

lrm_1.coef_
#prepare prediction result

prediction = lrm_1.predict(train_x)

i = 0

for v in prediction:

    if prediction[i] < 0:

        prediction[i] = 0

    i = i + 1
#visualize lrm_1

plt.figure(figsize=(5, 5))

plt.scatter(prediction, train_y)

plt.plot( [0,1000],[0,1000], color='red')

plt.xlim(-100, 1000)

plt.ylim(-100, 1000)

plt.xlabel('prediction')

plt.ylabel('train_y')

plt.title('Linear Regression Model')



print("RMSLE: ", rmsle(prediction, train_y))
#group data by month

train_data.datetime = train_data.datetime.apply(pd.to_datetime)

month = train_data.datetime.apply(lambda x : x.month)



plt.scatter(month.values, train_y)

plt.xlabel('month')

plt.ylabel('count')

plt.title('monthly count')
#group data by hours

hours = train_data.datetime.apply(lambda x : x.hour)



plt.scatter(hours.values, train_y)

plt.xlabel('hour')

plt.ylabel('count')

plt.title('hourly count')
#group by season

plt.scatter(train_data['season'].values, train_y)

plt.xlabel('season')

plt.ylabel('count')

plt.title('season')
#group by holiday

plt.scatter(train_data['holiday'].values, train_y)

plt.xlabel('holiday')

plt.ylabel('count')

plt.title('holiday')
#group by workingday

plt.scatter(train_data['workingday'].values, train_y)

plt.xlabel('workingday')

plt.ylabel('count')

plt.title('workingday')
#group by weather

plt.scatter(train_data['weather'].values, train_y)

plt.xlabel('weather')

plt.ylabel('count')

plt.title('weather')
#group by temperature

plt.scatter(train_data['temp'].values, train_y)

plt.xlabel('temperature')

plt.ylabel('count')

plt.title('temperature')
#group by "feels like" temperature

plt.scatter(train_data['atemp'].values, train_y)

plt.xlabel('"feels like" temperature')

plt.ylabel('count')

plt.title('"feels like" temperature')
#group by humidity

plt.scatter(train_data['humidity'].values, train_y)

plt.xlabel('humidity')

plt.ylabel('count')

plt.title('humidity')
#group by windspeed

plt.scatter(train_data['windspeed'].values, train_y)

plt.xlabel('windspeed')

plt.ylabel('count')

plt.title('windspeed')
#construct linear regression model 2 according to hours

train_hour = train_data

train_hour['hour'] = train_hour.datetime.apply(lambda x : x.hour)



train_hour_1 = train_hour.loc[train_hour['hour'] < 5]

x_hour_1 = train_hour_1.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)

                             

train_hour_2 = train_hour.loc[(train_hour['hour'] >= 5) & (train_hour['hour'] < 9)]

x_hour_2 = train_hour_2.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)

                             

train_hour_3 = train_hour.loc[(train_hour['hour'] >= 9) & (train_hour['hour'] < 13)]

x_hour_3 = train_hour_3.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)

                             

train_hour_4 = train_hour.loc[(train_hour['hour'] >= 13) & (train_hour['hour'] < 17)]

x_hour_4 = train_hour_4.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)

                             

train_hour_5 = train_hour.loc[train_hour['hour'] >= 17]

x_hour_5 = train_hour_5.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)



lrm_hour1 = linear_model.LinearRegression().fit(x_hour_1, train_hour_1['count'].values)

lrm_hour2 = linear_model.LinearRegression().fit(x_hour_2, train_hour_2['count'].values)

lrm_hour3 = linear_model.LinearRegression().fit(x_hour_3, train_hour_3['count'].values)

lrm_hour4 = linear_model.LinearRegression().fit(x_hour_4, train_hour_4['count'].values)

lrm_hour5 = linear_model.LinearRegression().fit(x_hour_5, train_hour_5['count'].values)

lrm_hour5.coef_
#prepare prediction result

result_hour1 = lrm_hour1.predict(x_hour_1)

i = 0

for v in result_hour1:

    if result_hour1[i] < 0:

        result_hour1[i] = 0

    i = i + 1

result_hour2 = lrm_hour2.predict(x_hour_2)

i = 0

for v in result_hour2:

    if result_hour2[i] < 0:

        result_hour2[i] = 0

    i = i + 1

result_hour3 = lrm_hour3.predict(x_hour_3)

i = 0

for v in result_hour3:

    if result_hour3[i] < 0:

        result_hour3[i] = 0

    i = i + 1

result_hour4 = lrm_hour4.predict(x_hour_4)

i = 0

for v in result_hour4:

    if result_hour4[i] < 0:

        result_hour4[i] = 0

    i = i + 1

result_hour5 = lrm_hour5.predict(x_hour_5)

i = 0

for v in result_hour5:

    if result_hour5[i] < 0:

        result_hour5[i] = 0

    i = i + 1

#visualize lrm_hour1

plt.figure(figsize=(5, 5))

plt.scatter(result_hour1, train_hour_1['count'].values)

plt.plot( [0,200],[0,200], color='red')

plt.xlim(-50, 300)

plt.ylim(-50, 300)

plt.xlabel('result')

plt.ylabel('train_y')

plt.title('Linear Regression Model - lrm_hour1')

print("RMSLE: ", rmsle(result_hour1, train_hour_1['count'].values))
#visualize lrm_hour2

plt.figure(figsize=(5, 5))

plt.scatter(result_hour2, train_hour_2['count'].values)

plt.plot( [0,800],[0,800], color='red')

plt.xlim(-50, 900)

plt.ylim(-50, 900)

plt.xlabel('result')

plt.ylabel('train_y')

plt.title('Linear Regression Model - lrm_hour2')

print("RMSLE: ", rmsle(result_hour2, train_hour_2['count'].values))
#visualize lrm_hour3

plt.figure(figsize=(5, 5))

plt.scatter(result_hour3, train_hour_3['count'].values)

plt.plot( [0,800],[0,800], color='red')

plt.xlim(-50, 900)

plt.ylim(-50, 900)

plt.xlabel('result')

plt.ylabel('train_y')

plt.title('Linear Regression Model - lrm_hour3')

print("RMSLE: ", rmsle(result_hour3, train_hour_3['count'].values))
#visualize lrm_hour4

plt.figure(figsize=(5, 5))

plt.scatter(result_hour4, train_hour_4['count'].values)

plt.plot( [0,800],[0,800], color='red')

plt.xlim(-50, 900)

plt.ylim(-50, 900)

plt.xlabel('result')

plt.ylabel('train_y')

plt.title('Linear Regression Model - lrm_hour4')

print("RMSLE: ", rmsle(result_hour4, train_hour_4['count'].values))
#visualize lrm_hour5

plt.figure(figsize=(5, 5))

plt.scatter(result_hour5, train_hour_5['count'].values)

plt.plot( [0,900],[0,900], color='red')

plt.xlim(-50, 1000)

plt.ylim(-50, 1000)

plt.xlabel('result')

plt.ylabel('train_y')

plt.title('Linear Regression Model - lrm_hour5')

print("RMSLE: ", rmsle(result_hour5, train_hour_5['count'].values))
#RandomForest

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor().fit(train_x, train_y)

prediction_rfr = rfr.predict(train_x)



#visualize

plt.figure(figsize=(5, 5))

plt.scatter(prediction_rfr, train_y)

plt.plot( [0,1000],[0,1000], color='red')

plt.xlim(-100, 1000)

plt.ylim(-100, 1000)

plt.xlabel('prediction')

plt.ylabel('train_y')

plt.title('Random Forest Regressor Model')



print("RMSLE: ", rmsle(prediction_rfr, train_y))