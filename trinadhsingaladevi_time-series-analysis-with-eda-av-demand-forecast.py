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
train = pd.read_csv('/kaggle/input/janatahack-demand-forecasting-analytics-vidhya/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-demand-forecasting-analytics-vidhya/test.csv')
print(train.shape)

train.head()
train.info()
train.describe()
train.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



sns.distplot(train["units_sold"])
print("skewness = ", train['units_sold'].skew())
cor = train.corr()

plt.figure(figsize= (15,12))

sns.heatmap(cor, annot = True,)
cont_feature = ['total_price','base_price']

cat_feature = ['is_featured_sku','is_display_sku']

fig,ax = plt.subplots(1,2,figsize = (18,5))

i = 221

for f in cont_feature:

    plt.subplot(i)

    sns.distplot(train[f])

    i += 1

# #     plt.subplot(i)

# #     train[f].plot.bar()

#     i += 1



#train['total_price'].plot.bar()

    
fig,ax = plt.subplots(1,2,figsize=(18,5))

i = 121

for f in cat_feature:

    plt.subplot(i)

    train[f].value_counts().plot.bar(title = f)

    #plt.hist(train[f])

    i += 1
fig,ax = plt.subplots(1,2,figsize=(18,5))

i = 121

for f in cat_feature:

    plt.subplot(i)

    train.groupby(f)['units_sold'].mean().plot.bar()

    i += 1


fig,ax = plt.subplots(1,2,figsize=(18,5))

i = 121

for f in cont_feature:

    plt.subplot(i)

    plt.scatter(train[f],train['units_sold'],label = f)

    plt.xlabel(f)

    plt.ylabel("units_sold")

    i += 1
#will see what is that highest outliered row



train[train["units_sold"] == train["units_sold"].max()]

#May be because of discount
train['week']= pd.to_datetime(train['week'])

train.groupby('week').sum()['units_sold'].plot(figsize = (20,8))

plt.xlabel("Week")

plt.ylabel("Units Sold")
train['year'] = pd.DatetimeIndex(train['week']).year

train.groupby('year').sum()['units_sold'].plot(figsize = (20,8))

test['year'] = pd.DatetimeIndex(test['week']).year
train['month'] = pd.DatetimeIndex(train['week']).month

test['month'] = pd.DatetimeIndex(test['week']).month

train.groupby('month').sum()['units_sold'].plot(figsize = (20,8))
train.groupby(['year','month']).sum()['units_sold'].plot(figsize = (20,8))
train['discount'] = train['base_price'] - train['total_price']

test['discount'] = test['base_price'] - test['total_price']

plt.scatter(train['discount'],train['units_sold'],label = 'discount')
train[train['total_price'].isnull() == True]
train["total_price"].fillna(train[train['sku_id']== 245338]['total_price'].mean(),inplace = True)
train.isnull().sum()
test.isnull().sum()
train['units_sold_log'] = np.log(train['units_sold'])

train['units_sold_log'].hist(bins=20) 

#test['units_sold_log'] = np.log(test['units_sold'])
train['total_price_log'] = np.log(train['total_price'])

train['total_price_log'].hist(bins=20) 

test['total_price_log'] = np.log(test['total_price'])
train['base_price_log'] = np.log(train['base_price'])

train['base_price_log'].hist(bins=20) 

test['base_price_log'] = np.log(test['base_price'])
test.head()
from sklearn.linear_model import LinearRegression



lr = LinearRegression()
from datetime import datetime    # To access datetime 

from pandas import Series

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd
train = pd.read_csv('/kaggle/input/janatahack-demand-forecasting-analytics-vidhya/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-demand-forecasting-analytics-vidhya/test.csv')

train_original=train.copy() 

test_original=test.copy()
train['week']= pd.to_datetime(train['week'])

test['week']= pd.to_datetime(test['week'])

train_original['week']= pd.to_datetime(train_original['week'])

test_original['week']= pd.to_datetime(test_original['week'])
for i in (train, test, test_original, train_original):

    i['year']=i['week'].dt.year 

    i['month']=i['week'].dt.month

    i['week_number']=i['week'].dt.week

    
train.index = train['week'] # indexing the Datetime to get the time period on the x-axis. 

df=train.drop('record_ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['units_sold'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Units Sold') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Units Sold") 

plt.legend(loc='best')
train.groupby('year')['units_sold'].mean().plot.bar()
train.groupby('month')['units_sold'].mean().plot.bar()
temp=train.groupby(['year', 'month'])['units_sold'].mean() 

temp.plot(figsize=(15,5), title= 'Units Sold (Monthwise)', fontsize=14)

train.Timestamp = pd.to_datetime(train["week"]) 

train.index = train.Timestamp 



weekly = train.resample('W').mean() 

# # Converting to monthly mean 

monthly = train.resample('M').mean()
weekly["units_sold"].plot(figsize=(15,8), title= 'Weekly', fontsize=14, ) 

monthly["units_sold"].plot(figsize=(15,8), title= 'Monthly', fontsize=14) 
train["week"].sort_values(ascending = False)
Train=train.loc['2011-01-08':'2013-09-03'] 

valid=train.loc['2013-09-04':'2013-12-03']
type(Train["units_sold"])
Train['units_sold'].plot(figsize=(15,8), title= 'units sold', fontsize=14, label='train') 

valid['units_sold'].plot(figsize=(15,8), title= 'units sold', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("units sold") 

plt.legend(loc='best') 

plt.show()
y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['units_sold'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['units_sold'], label='Train') 

plt.plot(valid['units_sold'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['units_sold'].rolling(20).mean().iloc[-1] # average of last 20 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['units_sold'], label='Train') 

plt.plot(valid['units_sold'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['units_sold'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['units_sold'], label='Train') 

plt.plot(valid['units_sold'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_squared_log_error as msle

from math import sqrt



rms = sqrt(mean_squared_error(valid.units_sold, y_hat_avg.moving_avg_forecast)) 

print("rms: ",rms)

rmsle = sqrt(msle(valid.units_sold, y_hat_avg.moving_avg_forecast)) 

print("rmsle: ",rmsle)
import numpy as np

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

y_hat_avg = valid.copy() 

fit2 = SimpleExpSmoothing(np.asarray(Train['units_sold'])).fit(smoothing_level=0.6,optimized=False) 

y_hat_avg['SES'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['units_sold'], label='Train') 

plt.plot(valid['units_sold'], label='Valid') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.units_sold, y_hat_avg.SES)) 

print(rms)

rmsle = sqrt(msle(valid.units_sold, y_hat_avg.SES)) 

print(rmsle)
Train.index
import statsmodels.api as sm 

plt.figure(figsize=(16,8)) 

sm.tsa.seasonal_decompose(Train["units_sold"],period = 3).plot() 

result = sm.tsa.stattools.adfuller(train.units_sold) 

plt.show()
y_hat_avg = valid.copy() 

fit1 = Holt(np.asarray(Train['units_sold'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['units_sold'], label='Train') 

plt.plot(valid['units_sold'], label='Valid') 

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.units_sold, y_hat_avg.Holt_linear)) 

print(rms)