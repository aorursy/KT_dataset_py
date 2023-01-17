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
# load libraries 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import show

import plotly.express as px



# combine and create single dataframe

chicago_df_1 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv', error_bad_lines=False)

chicago_df_2 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)

chicago_df_3 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)

chicago_df_4 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)
# combining the datasets

df = pd.concat([chicago_df_1,chicago_df_2,chicago_df_3,chicago_df_4],ignore_index=False,axis=0)
df.shape
# let's view the head of the training dataset

df.head()
# select only the necessary columns

df = df[['ID','Date','Primary Type','Location Description','Arrest','Domestic']]
df.head()
df.info()
# change the column date dtype from object to date

df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date 

df.index = pd.DatetimeIndex(df.Date)
# verify the change

df.head()
# get the summary

print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n\n" ,df.columns.tolist())

print ("\nMissing values : \n\n", df.isnull().any())

print ("\nUnique values :  \n\n",df.nunique())
# Examine the null records of Location Description

df[df["Location Description"].isnull()]

# drop these records

df = df.dropna()

# print the count of Null records in each column

df.isnull().sum()
# set figure size

plt.figure(figsize = (15, 10))



# plot the records

ax=sns.countplot(x= 'Primary Type', data = df, order = df['Primary Type'].value_counts().iloc[:10].index, palette = 'RdBu_r')



# set individual bar lables using above list

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x(), i.get_height(),

            str(i.get_height()), fontsize=15,

color='dimgrey')

show()

# set the figure size

plt.figure(figsize = (15, 10))



# plot the values

ax = sns.countplot(y= 'Location Description', data = df, order = df['Location Description'].value_counts().iloc[:15].index,palette = 'RdBu_r')



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.3, i.get_y()+.5, 

            str(i.get_width()), fontsize=15,

color='dimgrey')

show()

# Resample is a Convenience method for frequency conversion and resampling of time series.



# resample into Years



plt.plot(df.resample('Y').size())

plt.title('Crimes Count Per Year')

plt.xlabel('Years')

plt.ylabel('Number of Crimes')
# resample into Months



plt.plot(df.resample('M').size())

plt.title('Crimes Count Per Month')

plt.xlabel('Months')

plt.ylabel('Number of Crimes')
# aggregating the number of cases per month for all years

ts_df = pd.DataFrame(df.resample('M').size().reset_index())

ts_df.columns = ['Date', 'Crime Count'] # renaming the columns
ts_df.head()
# plot interactive slider chart

fig = px.line(ts_df, x='Date',y='Crime Count', title= 'Crime count')



fig.update_xaxes(

rangeslider_visible =True,

rangeselector=dict(

        buttons=list([

                dict(count=1,label="1y",step="year",stepmode="backward"),

                dict(count=2,label="3y",step="year",stepmode="backward"),

                dict(count=3,label="5y",step="year",stepmode="backward"),

                dict(step="all")

                    ])

                )

                )

fig.show()
ts_df = ts_df.set_index('Date')
# splitting into train and test set

train = ts_df[:181]

test = ts_df[181:]

print(train.shape)

print(test.shape)
plt.plot(train)

plt.plot(test)
from fbprophet import Prophet
# creating the dataframe

prophet_df = train.reset_index()

prophet_df .head()
prophet_df = prophet_df.rename(columns= {'Date':'ds','Crime Count':'y'})

prophet_df.head()
m = Prophet()

m.fit(prophet_df)

# Forcasting into the future

future = m.make_future_dataframe(periods=12, freq='M')

forecast = m.predict(future)
forecast.head()
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
plot = m.plot_components(forecast)
forecast_df = pd.DataFrame(forecast)

forecast_df.head()
# preparing the dataframe with date and forecast

forecast_df=forecast_df[['ds','yhat']]

forecast_df=forecast_df.set_index('ds')

forecast_df.head()
# plot the predictions

plt.figure(figsize=(20,10))

plt.plot(train, label ='Train')

plt.plot(test, label='Test')

plt.plot(forecast_df, label='Forecast')

plt.xlabel('Year')

plt.ylabel('Count')

plt.title('Plotting Train vs Test vs Predicted Crime rate')

plt.legend()

plt.show()
test['Crime Count']
forecast_df['yhat'][181:]
# calculate error

test['fbprophet_error'] = test['Crime Count'] - forecast_df['yhat'][181:]



rmse = np.sqrt(np.mean(test.fbprophet_error**2)).round(2)

mape = np.round(np.mean(np.abs(100*(test.fbprophet_error/test['Crime Count'])), 0))



print('RMSE = $', rmse)

print('MAPE =', mape, '%')
!pip install pmdarima
import pmdarima as pm
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(train)
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(train,lags=20)
model = pm.auto_arima(train,m=12,start_p=0,start_q=3, max_order=5, 

                      error_action='ignore',test='adf',seasonal=True,

                      trace=True,stepwise=True)
model.summary()
# create a dataframe with test date index

prediction = pd.DataFrame(model.predict(n_periods=12),index = test.index,columns =['Predicted Crime Count'])
# print predictions

prediction
test
# plot the predictions

plt.plot(train, label ='Train')

plt.plot(test['Crime Count'], label='Test')

plt.plot(prediction, label='Prediction')

plt.xlabel('Year')

plt.ylabel('Count')

plt.title('Plotting Train vs Test vs Predicted Crime rate')

plt.legend()

plt.show()
# calculate error

test['arima_error'] = test['Crime Count'] - prediction['Predicted Crime Count']



rmse = np.sqrt(np.mean(test.arima_error**2)).round(2)

mape = np.round(np.mean(np.abs(100*(test.arima_error/test['Crime Count'])), 0))



print('RMSE = $', rmse)

print('MAPE =', mape, '%')
out = model.plot_diagnostics()