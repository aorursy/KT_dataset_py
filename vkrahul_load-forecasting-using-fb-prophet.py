# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
from fbprophet import Prophet
def mean_abs_percentage_error(y_true,y_pred):

    y_true,y_pred = np.array(y_true),np.array(y_pred)

    return np.abs((y_true-y_pred)/y_true)
df = pd.read_csv('../input/energy-consumption-cleaned3.csv',index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
df[df['Date']==pd.datetime(2011,7,1)]
df.rename(columns={'Unnamed: 52':'Totals'},inplace=True)
df.head()
data = df[['Date','Totals']]
data.rename(columns={'Date':'ds','Totals':'y'},inplace=True)
from matplotlib.pylab import rcParams

#divide into train and validation set

train3 = data[(data.set_index('ds').index>=pd.datetime(2017,1,1)) & (data.set_index('ds').index<pd.datetime(2019,2,1))]

valid3 = data[data.set_index('ds').index>=pd.datetime(2019,2,1)]



rcParams['figure.figsize']=12,6



#plotting the data

plt.plot(train3['ds'],train3['y'])

plt.plot(valid3['ds'],valid3['y'])
from matplotlib.pylab import rcParams

#divide into train and validation set

train = data[(data.set_index('ds').index>=pd.datetime(2016,1,1)) & (data.set_index('ds').index<pd.datetime(2019,2,1))]

valid = data[data.set_index('ds').index>=pd.datetime(2019,2,1)]



rcParams['figure.figsize']=12,6



#plotting the data

plt.plot(train['ds'],train['y'])

plt.plot(valid['ds'],valid['y'])
holidays = pd.read_excel('../input/ukbankholidays.xls')
holidays.head()
holidays['holiday'] = 'BANK_HOLIDAYS'

holidays['lower_window'] = 0

holidays['upper_window'] = 0

holidays.rename(columns={'UK BANK HOLIDAYS':'ds'},inplace=True)
holidays.head()
train.head()
weather = pd.read_csv('../input/KEW_WEATHER.csv')

weather['DATE'] = pd.to_datetime(weather['DATE'])
weather.info()
temperature = weather[['DATE','TAVG']]

temperature.set_index('DATE',inplace=True)
precipitation = weather[['DATE','PRCP']]

precipitation.set_index('DATE',inplace=True)
train_temperature = temperature[(temperature.index>=pd.datetime(2016,1,1)) & (temperature.index<pd.datetime(2019,2,1))]

train_precipitation = precipitation[(precipitation.index>=pd.datetime(2016,1,1)) & (precipitation.index<pd.datetime(2019,2,1))]
test_temperature =temperature[(temperature.index>=pd.datetime(2019,2,1)) & (temperature.index<=pd.datetime(2019,2,28))]

test_precipitation =precipitation[(precipitation.index>=pd.datetime(2019,2,1)) & (precipitation.index<=pd.datetime(2019,2,28))]
dates=[]

values=[]

for i in train.set_index('ds').index:

    if(i not in train_temperature.index):

        print(i)

        dates.append(pd.to_datetime(i))

        values.append(50+np.random.randint(-2,2))
train_temperature2 = pd.concat([train_temperature,pd.DataFrame({'Date':dates,'TAVG':values}).set_index('Date')])
train_temperature.loc['2018-11-16 00:00:00']
train_temperature2.plot()
dates=[]

values=[]

for i in train.set_index('ds').index:

    if(i not in train_precipitation.index):

        print(i)

        dates.append(pd.to_datetime(i))

        values.append(0+(.01*np.random.randint(-12,12)))
# values
train_precipitation2 = pd.concat([train_precipitation,pd.DataFrame({'Date':dates,'PRCP':np.nan}).set_index('Date')])
train_precipitation2.loc['2018-11-17 00:00:00']
train_precipitation2 = train_precipitation2.interpolate(method='linear')

train_precipitation2.loc['2018-11-17 00:00:00']
train_precipitation2.plot()
for i in train_temperature2.index:



    if(i not in train.set_index('ds').index):

        print(i)

        train_temperature2.drop(index=i,inplace=True)
for i in train_precipitation2.index:



    if(i not in train.set_index('ds').index):

        print(i)

        train_precipitation2.drop(index=i,inplace=True)
train_temperature2.shape
train_precipitation2.shape
train_temperature3 = train_temperature2[(train_temperature2.index>=pd.datetime(2017,1,1)) & (train_temperature2.index<pd.datetime(2019,2,1))]

train_precipitation3 = train_precipitation2[(train_precipitation2.index>=pd.datetime(2017,1,1)) & (train_precipitation2.index<pd.datetime(2019,2,1))]
train2 = train.merge(train_temperature2.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)

train2 = train2.merge(train_precipitation2.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)
train3 = train3.merge(train_temperature3.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)

train3 = train3.merge(train_precipitation3.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)
help(Prophet.add_regressor)
train2.head()
model=Prophet(daily_seasonality=True,n_changepoints=37,holidays=holidays)

model.add_regressor('TAVG',prior_scale=0.5,mode='multiplicative')

model.add_regressor('PRCP',prior_scale=0.5,mode='multiplicative')

# model.add_country_holidays(country_name='UK')

# m.add_seasonality('self_define_cycle',period=8,fourier_order=8,mode='additive')
model3=Prophet(daily_seasonality=True,n_changepoints=37,holidays=holidays)

model3.add_regressor('TAVG',prior_scale=0.5,mode='multiplicative')

model3.add_regressor('PRCP',prior_scale=0.5,mode='multiplicative')

# model.add_country_holidays(country_name='UK')

# m.add_seasonality('self_define_cycle',period=8,fourier_order=8,mode='additive')
model.fit(train2)
model3.fit(train3)
model.train_holiday_names
future = model.make_future_dataframe(periods=28)

future.tail()
# import utils

# futures = utils.add_regressor_to_future(future, [test_temperature,test_precipitation])


future3 = model3.make_future_dataframe(periods=28)

future3.tail()
future['TAVG']=pd.concat([train_temperature2,test_temperature]).reset_index()['TAVG']

future['PRCP']=pd.concat([train_precipitation2,test_precipitation]).reset_index()['PRCP']

future3['TAVG']=pd.concat([train_temperature3,test_temperature]).reset_index()['TAVG']

future3['PRCP']=pd.concat([train_precipitation3,test_precipitation]).reset_index()['PRCP']
future3.isna().sum()
# Python

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Python

forecast3 = model3.predict(future3)

forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#plot the predictions for validation set

plt.plot(train2.set_index('ds')['y'], label='Train')

plt.plot(valid.set_index('ds'), label='Valid',alpha=0.5)

plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat'], label='Prediction',alpha=0.5)

# plt.legend()

# plt.save_fig('feb_forecast.png', bbox_inches='tight')
#plot the predictions for validation set

plt.plot(train3.set_index('ds')['y'], label='Train')

plt.plot(valid3.set_index('ds'), label='Valid',alpha=0.4)

plt.plot(forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat'], label='Prediction',alpha=0.4)

# plt.legend()

plt.show()
fig2 = model.plot_components(forecast)
fig4 = model.plot_components(forecast3)
fig, ax = plt.subplots()

plt.plot(valid.set_index('ds'), label='Valid',color='orange')

plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat'], label='Prediction',color='blue')

# upper=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

# lower=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

# ax.fill_between(valid.set_index('ds').index,forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'],

#                 forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], alpha=0.4)

plt.legend()

plt.savefig('three_years_train.png', bbox_inches='tight')
fig, ax = plt.subplots()

plt.plot(valid.set_index('ds'), label='Valid',color='orange')

plt.plot(forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat'], label='Prediction',color='blue')

# upper=plt.plot(forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

# lower=plt.plot(forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

# ax.fill_between(valid.set_index('ds').index,forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'],

#                 forecast3[forecast3['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], alpha=0.4)

plt.legend()

plt.savefig('two_years_train.png', bbox_inches='tight')
fig, ax = plt.subplots()

plt.plot(valid[(valid['ds']>=pd.datetime(2019,2,1)) & (valid['ds']<pd.datetime(2019,2,15))].set_index('ds'), label='Valid',color='orange')

plt.plot(forecast3[(forecast3['ds']>=pd.datetime(2019,2,1)) & (forecast3['ds']<pd.datetime(2019,2,15))].set_index('ds')['yhat'], label='Prediction',color='blue')

upper=plt.plot(forecast3[(forecast3['ds']>=pd.datetime(2019,2,1)) & (forecast3['ds']<pd.datetime(2019,2,15))].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

lower=plt.plot(forecast3[(forecast3['ds']>=pd.datetime(2019,2,1)) & (forecast3['ds']<pd.datetime(2019,2,15))].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

ax.fill_between(valid[(valid['ds']>=pd.datetime(2019,2,1)) & (valid['ds']<pd.datetime(2019,2,15))].set_index('ds').index,

                forecast3[(forecast3['ds']>=pd.datetime(2019,2,1)) & (forecast3['ds']<pd.datetime(2019,2,15))].set_index('ds')['yhat_upper'],

                forecast3[(forecast3['ds']>=pd.datetime(2019,2,1)) & (forecast3['ds']<pd.datetime(2019,2,15))].set_index('ds')['yhat_lower'], 

                alpha=0.5)

plt.legend()

plt.savefig('two_years_train_week.png', bbox_inches='tight')
# valid[valid['ds']>=pd.datetime(2019,1,28)].set_index('ds')['y'].values
# forecast[forecast['ds']>=pd.datetime(2019,1,28)].set_index('ds')['yhat'].values

# valid.set_index('ds')
#calculate rmse

from math import sqrt

from sklearn.metrics import mean_squared_error



rms = sqrt(mean_squared_error(valid.set_index('ds'),forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat']))

print(rms)
np.mean(mean_abs_percentage_error(valid[valid['ds']>=pd.datetime(2019,2,1)].set_index('ds')['y'],forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat']))*100
train2.head()
total_temperature=pd.concat([train_temperature2,test_temperature]).reset_index()

total_precipitation=pd.concat([train_precipitation2,test_precipitation]).reset_index()
total_df = pd.concat([forecast.reset_index()['ds'],forecast['yhat'],data['y'][:1154]],axis=1)
total_df
total_df.loc[:,['y','yhat']].corr()
from scipy.stats import skew
f, ax = plt.subplots(figsize=(8,8))

sns.distplot((total_df.loc['2019':,'yhat'] - total_df.loc['2019':,'y']), ax=ax, color='0.4')

ax.grid(ls=':')

ax.set_xlabel('residuals', fontsize=15)

ax.set_ylabel("normalised frequency", fontsize=15)

ax.grid(ls=':')



[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];



ax.text(0.05, 0.9, "Skewness = {:+4.2f}\nMedian = {:+4.2f}".\

        format(skew(total_df.loc['2019':,'yhat'] - total_df.loc['2019':,'y']), (total_df.loc['2019':,'yhat'] - total_df.loc['2019':,'y']).median()), \

        fontsize=14, transform=ax.transAxes)



ax.axvline(0, color='0.4')



ax.set_title('Residuals distribution (test set)', fontsize=17)
corr = total_df.set_index('ds').loc[:,['y','yhat']].rolling(window=30, center=True).corr().iloc[0::2,1]
corr.index = corr.index.droplevel(1)
%matplotlib inline

f, ax = plt.subplots(figsize=(14, 8))



corr.plot(ax=ax, lw=3, color='0.4')



ax.axhline(0.8, color='0.8', zorder=-1)

ax.axhline(0.6, color='0.8', zorder=-1)

ax.axvline('2019', color='k', zorder=-1)

ax.grid(ls=':')

# ax.set_ylim([0.5, 0.9])

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel("Pearson's R", fontsize=15)

ax.grid(ls=':')

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]



ax.set_yticks(np.arange(0.5, 1., 0.1)); 



ax.set_title('30 days running window correlation\nbetween observed and modelled / predicted values', fontsize=15)

# total_temperature=pd.concat([train_temperature2,test_temperature]).reset_index()

# total_precipitation=pd.concat([train_precipitation2,test_precipitation]).reset_index()
train
from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(model, initial='730 days', period='30 days', horizon = '31 days')

df_cv.head()
df_cv
corr = df_cv.set_index('ds').loc[:,['y','yhat']].rolling(window=30, center=True).corr().iloc[0::2,1]
corr.index = corr.index.droplevel(1)
%matplotlib inline

f, ax = plt.subplots(figsize=(14, 8))



corr.plot(ax=ax, lw=3, color='0.4')



ax.axhline(0.8, color='0.8', zorder=-1)

ax.axhline(0.6, color='0.8', zorder=-1)

ax.axvline('2018/12/1', color='k', zorder=-1)

ax.grid(ls=':')

# ax.set_ylim([0.5, 0.9])

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel("Pearson's R", fontsize=15)

ax.grid(ls=':')

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]



ax.set_yticks(np.arange(0.5, 1., 0.1)); 



ax.set_title('30 days running window correlation\nbetween observed and modelled / predicted values', fontsize=15)

df_cv.set_index('ds',inplace=True)
corr_season_test = df_cv.loc['2018':,['y','yhat']].groupby(df_cv.loc['2018':,:].index.month).corr()

corr_season_train = df_cv.loc[:'2018',['y','yhat']].groupby(df_cv.loc[:'2018',:].index.month).corr()

corr_season = df_cv.loc[:,['y','yhat']].groupby(df_cv.loc[:,:].index.month).corr()
f, ax = plt.subplots(figsize=(8,8))

corr_season_train.xs('y', axis=0, level=1)['yhat'].plot(ax=ax, lw=3, marker='o', markersize=12, label='train set', ls='-', color='k',alpha=0.4)

corr_season_test.xs('y', axis=0, level=1)['yhat'].plot(ax=ax, lw=3, marker='o', markersize=12, label='test set', ls='--', color='k')

# corr_season.xs('y', axis=0, level=1)['yhat'].plot(ax=ax, lw=3, marker='o', markersize=12)



ax.legend(fontsize=17, loc=3)



ax.set_xticks(range(1, 13))

ax.set_xticklabels(list('JFMAMJJASOND'))

ax.set_xlabel('month', fontsize=15)

ax.set_ylabel("Pearson's R", fontsize=15)

ax.grid(ls=':')

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]

[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]



ax.set_title('correlation per month', fontsize=17)

from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.head()
# Python

from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')

fig, ax = plt.subplots()

plt.plot(df_cv.set_index('ds')['y'], label='Valid',color='orange')

plt.plot(df_cv.set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.7)

upper=plt.plot(df_cv.set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

lower=plt.plot(df_cv.set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

ax.fill_between(df_cv.set_index('ds').index,df_cv.set_index('ds')['yhat_upper'],

                df_cv.set_index('ds')['yhat_lower'], alpha=0.4)

plt.legend()

# plt.show()

plt.savefig('cross_val.png', bbox_inches='tight')

# plt.plot(, label='Prediction',alpha=0.7,color='green')

# plt.show()
fig, ax = plt.subplots()

plt.plot(df_cv.set_index('ds')['y'], label='Valid',color='orange',alpha=0.6)

plt.plot(df_cv.set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.7)

plt.legend()

plt.savefig('cross_val2.png', bbox_inches='tight')

# plt.plot(, label='Prediction',alpha=0.7,color='green')

# plt.show()
fig, ax = plt.subplots()

plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['y'], label='Valid',color='orange',alpha=0.6)

plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.7)

upper=plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

lower=plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

ax.fill_between(df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds').index,

                df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['yhat_upper'],

                df_cv[(df_cv['ds']>=pd.datetime(2018,1,1)) & (df_cv['ds']<pd.datetime(2018,2,1))].set_index('ds')['yhat_lower'], 

                alpha=0.4)

plt.legend()

plt.savefig('cross_val_month.png', bbox_inches='tight')

# plt.plot(, label='Prediction',alpha=0.7,color='green')

# plt.show()
fig, ax = plt.subplots()

plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['y'], label='Valid',color='orange',alpha=0.6)

plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.7)

upper=plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)

lower=plt.plot(df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')

ax.fill_between(df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds').index,

                df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['yhat_upper'],

                df_cv[(df_cv['ds']>=pd.datetime(2018,1,7)) & (df_cv['ds']<pd.datetime(2018,1,14))].set_index('ds')['yhat_lower'], 

                alpha=0.4)

plt.legend()

plt.savefig('cross_val_week.png', bbox_inches='tight')

# plt.plot(, label='Prediction',alpha=0.7,color='green')

# plt.show()
train2_dec = train2[(train2['ds']<pd.datetime(2018,12,1)) & (train2['ds']>=pd.datetime(2016,1,1))]

valid2_dec=train2[(train2['ds']<pd.datetime(2019,1,1)) & (train2['ds']>=pd.datetime(2018,12,1))]

temperature_dec = total_temperature[(total_temperature['index']<pd.datetime(2019,1,1)) & (total_temperature['index']>=pd.datetime(2016,1,1))]

precipitation_dec = total_precipitation[(total_precipitation['index']<pd.datetime(2019,1,1)) & (total_precipitation['index']>=pd.datetime(2016,1,1))]
model2=Prophet(daily_seasonality=True,n_changepoints=49,holidays=holidays)

model2.add_regressor('TAVG',prior_scale=5,standardize=False)

model2.add_regressor('PRCP',prior_scale=5,standardize=False)

# Python

model2.fit(train2_dec)

future2 = model2.make_future_dataframe(periods=31)

future2.set_index('ds',inplace=True) 

future2['TAVG']=temperature_dec.set_index('index')['TAVG']

future2['PRCP']=precipitation_dec.set_index('index')['PRCP']

forecast2 = model2.predict(future2.reset_index())

forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model2.plot(forecast2)
np.mean(mean_abs_percentage_error(valid2_dec[(valid2_dec['ds']>=pd.datetime(2018,12,1)) & (valid2_dec['ds']<pd.datetime(2018,12,8))].set_index('ds')['y'],forecast2[(forecast2['ds']>=pd.datetime(2018,12,1)) & ((forecast2['ds']<pd.datetime(2018,12,8)))].set_index('ds')['yhat']))
rms = sqrt(mean_squared_error(valid2_dec.set_index('ds')['y'],forecast2[forecast2['ds']>=pd.datetime(2018,12,1)].set_index('ds')['yhat']))

print(rms)
fig, ax = plt.subplots()

plt.plot(valid2_dec.set_index('ds')['y'], label='Valid',color='orange')

plt.plot(forecast2[forecast2['ds']>=pd.datetime(2018,12,1)].set_index('ds')['yhat'], label='Prediction',color='blue')