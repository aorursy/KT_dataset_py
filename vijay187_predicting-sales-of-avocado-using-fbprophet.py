# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

data.head(10)
data.info()
data.isnull().sum()
data['Date'] = pd.to_datetime(data['Date'])
data.type.value_counts()
data.region.nunique()

data.region.value_counts().sort_index()
sales = data[(data['region']=='Albany')&(data['type']=='organic')][['Date','Total Bags']]
sales = sales.sort_values(by='Date')
sales = sales.rename(columns = {'Date':'ds','Total Bags':'y'})

sales = sales.reset_index(drop=True)

sales.head(6)
#plot daily sales

pd.plotting.register_matplotlib_converters()

ax = sales.set_index('ds').plot(figsize=(12,4))

ax.set_ylabel('Daily number of sales')

ax.set_xlabel('Date')

plt.show()
import pandas as pd

holidays = pd.read_csv('/kaggle/input/holidays/us-federal-holidays-2011-2020.csv')
holidays.head()
holidays.Date = pd.to_datetime(holidays.Date)
holidays = holidays.rename(columns={'Date':'ds','Holiday':'holiday'})

holidays = holidays[['holiday','ds']]

holidays.head()
from fbprophet import Prophet



#set uncertanity interval to 95%

mymodel = Prophet(interval_width  = 0.95, holidays = holidays)

mymodel.fit(sales)



# dataframe that extends into future 6 weeks 

future_dates = mymodel.make_future_dataframe(periods = 6*7)



print("First week to forecast.")

future_dates.tail(7)
#Predictions

forecast = mymodel.predict(future_dates)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
mymodel.plot(forecast);
mymodel.plot_components(forecast);
data['region'].unique()
print('Enter region: ')

region = input()

print('\n')

print('choose conventional/organic: ')

type_ = input()



sales = data[(data['region']==region)&(data['type']==type_)][['Date','Total Bags']]

sales = sales.sort_values(by='Date')



sales = sales.rename(columns = {'Date':'ds','Total Bags':'y'})

sales = sales.reset_index(drop=True)



pd.plotting.register_matplotlib_converters()

ax = sales.set_index('ds').plot(figsize=(12,4))

ax.set_ylabel('Daily number of sales')

ax.set_xlabel('Date')

plt.show()

mymodel = Prophet(interval_width  = 0.95, holidays = holidays)

mymodel.fit(sales)



future_dates = mymodel.make_future_dataframe(periods = 6*7)

forecast = mymodel.predict(future_dates)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)



print('FORECAST MODELS')

mymodel.plot(forecast);

mymodel.plot_components(forecast);