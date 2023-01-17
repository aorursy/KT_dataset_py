# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
from fbprophet import Prophet
def mean_abs_percentage_error(y_true,y_pred):

    y_true,y_pred = np.array(y_true),np.array(y_pred)

    return np.abs((y_true-y_pred)/y_true)
df = pd.read_table('../input/random-load-data/data.txt',header=None)
df.shape
df.columns=['y']
df.head()
df['Date'] = pd.date_range(pd.datetime(2017,1,1),pd.datetime(2018,12,31))
df.tail()
df.head()
data=df
data.rename(columns={'Date':'ds'},inplace=True)
from matplotlib.pylab import rcParams

#divide into train and validation set

train = data[(data.set_index('ds').index>=pd.datetime(2017,1,1)) & (data.set_index('ds').index<pd.datetime(2018,10,1))]

valid = data[data.set_index('ds').index>=pd.datetime(2018,10,1)]



rcParams['figure.figsize']=12,6



#plotting the data

plt.plot(train['ds'],train3['y'])

plt.plot(valid['ds'],valid3['y'])
holidays = pd.read_excel('../input/the-national-archives-in-kew-uk/ukbankholidays.xls')
holidays.head()
holidays['holiday'] = 'national'

holidays['lower_window'] = 0

holidays['upper_window'] = 0

holidays.rename(columns={'UK BANK HOLIDAYS':'ds'},inplace=True)
holidays.head()
train.head()
weather = pd.read_csv('../input/the-national-archives-in-kew-uk/KEW_WEATHER.csv')

weather['DATE'] = pd.to_datetime(weather['DATE'])
temperature = weather[['DATE','TAVG']]

temperature.set_index('DATE',inplace=True)
train_weather = temperature[(temperature.index>=pd.datetime(2016,1,1)) & (temperature.index<pd.datetime(2019,2,1))]
test_weather =temperature[(temperature.index>=pd.datetime(2019,2,1)) & (temperature.index<=pd.datetime(2019,2,28))]
dates=[]

values=[]

for i in train.set_index('ds').index:

    if(i not in train_weather.index):

        print(i)

        dates.append(pd.to_datetime(i))

        values.append(50+np.random.randint(-2,2))
train_weather2 = pd.concat([train_weather,pd.DataFrame({'Date':dates,'TAVG':values}).set_index('Date')])
for i in train_weather2.index:

    if(i not in train.set_index('ds').index):

        print(i)

        train_weather2.drop(index=i,inplace=True)
train_weather2.shape
train_weather3 = train_weather2[(train_weather2.index>=pd.datetime(2017,1,1)) & (train_weather2.index<pd.datetime(2019,2,1))]
train2 = train.merge(train_weather2.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)
train3 = train3.merge(train_weather3.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)
help(Prophet.add_regressor)
model=Prophet()

# model.add_regressor('TAVG',prior_scale=5,standardize=False)

# model.add_country_holidays(country_name='UK')

# m.add_seasonality('self_define_cycle',period=8,fourier_order=8,mode='additive')
model.fit(train)
future = model.make_future_dataframe(periods=92)

future.tail()

# Python

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#plot the predictions for validation set

plt.plot(train.set_index('ds'), label='Train')

plt.plot(valid.set_index('ds'), label='Valid')

plt.plot(forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat'], label='Prediction',alpha=0.4)

# plt.legend()

# plt.save_fig('feb_forecast.png', bbox_inches='tight')
fig2 = model.plot_components(forecast)
fig, ax = plt.subplots()

plt.plot(valid.set_index('ds'), label='Valid',color='orange')

plt.plot(forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.4)

# upper=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

# lower=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

# ax.fill_between(valid.set_index('ds').index,forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'],

#                 forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], alpha=0.4)

plt.legend()

plt.savefig('three_years_train.png', bbox_inches='tight')
# forecast[forecast['ds']>=pd.datetime(2019,1,28)].set_index('ds')['yhat'].values

# valid.set_index('ds')
#calculate rmse

from math import sqrt

from sklearn.metrics import mean_squared_error



rms = sqrt(mean_squared_error(valid.set_index('ds'),forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat']))

print(rms)
np.mean(mean_abs_percentage_error(valid[valid['ds']>=pd.datetime(2018,10,1)].set_index('ds')['y'],forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat']))*100