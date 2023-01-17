# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")



# loading packages

# basic + dates 

import numpy as np

import pandas as pd

from pandas import datetime



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns # advanced vizs

%matplotlib inline



# statistics

#https://towardsdatascience.com/what-why-and-how-to-read-empirical-cdf-123e2b922480

from statsmodels.distributions.empirical_distribution import ECDF



# time series analysis

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# prophet by Facebook

from fbprophet import Prophet
# importing train data to learn

train = pd.read_csv("../input/rossmann-store-sales/train.csv", 

                    parse_dates = True, low_memory = False, index_col = 'Date')



# additional store data

store = pd.read_csv("../input/rossmann-store-sales/store.csv", 

                    low_memory = False)

# time series as indexes

train.index
print(train.shape)

train.head()
# data extraction

train['Year'] = train.index.year

train['Month'] = train.index.month

train['Day'] = train.index.day

train['WeekOfYear'] = train.index.weekofyear



# adding new variable

train['SalePerCustomer'] = train['Sales']/train['Customers']

train['SalePerCustomer'].describe()
sns.set(style = "ticks")# to format into seaborn 

c = '#386B7F' # basic color for plots

plt.figure(figsize = (12, 6))



plt.subplot(311)

cdf = ECDF(train['Sales'])

plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);

plt.xlabel('Sales'); plt.ylabel('ECDF');



# plot second ECDF  

plt.subplot(312)

cdf = ECDF(train['Customers'])

plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);

plt.xlabel('Customers');



# plot second ECDF  

plt.subplot(313)

cdf = ECDF(train['SalePerCustomer'])

plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);

plt.xlabel('Sale per Customer');
# closed stores

train[(train.Open == 0) & (train.Sales == 0)].head()
train[(train.Open==0)].shape
# opened stores with zero sales

zero_sales = train[(train.Open != 0) & (train.Sales == 0)]

print("In total: ", zero_sales.shape)

zero_sales.head(5)
print("Closed stores and days which didn't have any sales won't be counted into the forecasts.")

train = train[(train["Open"] != 0) & (train['Sales'] != 0)]



print("In total: ", train.shape)
# additional information about the stores

store.head()
#missing values

store.isnull().sum()
# missing values in copetitin distance 

store[pd.isnull(store.CompetitionDistance)]
# fill NaN with a median value (skewed distribuion)

store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
store[pd.isnull(store.Promo2SinceWeek)]

# filling NAN value with zero 

store.fillna(0, inplace = True)
print("Joining train set with an additional store information.")



# by specifying inner join we make sure that only those observations 

# that are present in both train and store sets are merged together

train_store = pd.merge(train, store, how = 'inner', on = 'Store')



print("In total: ", train_store.shape)

train_store.head()
train_store.groupby('StoreType')['Sales'].describe()
train_store.groupby('StoreType')['Customers', 'Sales'].sum()
# sales trends

sns.factorplot(data = train_store, x = 'Month', y = "Sales", 

               palette = 'plasma',

               row = 'Promo',

               hue = 'StoreType',

               color = c)
# sales trends

sns.factorplot(data = train_store, x = 'Month', y = "Customers", 

               palette = 'plasma',

               col = 'Promo',

               hue = 'StoreType',

               color = c) 
# sale per customer trends

sns.factorplot(data = train_store, x = 'Month', y = "SalePerCustomer", 

               palette = 'plasma',

               hue = 'StoreType',

               col = 'Promo', # per promo in the store in rows

               color = c) 
# stores which are opened on Sundays

train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].unique()
 #competition open time (in months)

train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + (train_store.Month - train_store.CompetitionOpenSinceMonth)

# Promo open time

# to convert weeks into momths we divided by 4

train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) +(train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

# replace NA's by 0

train_store.fillna(0, inplace = True)

# average PromoOpen time and CompetitionOpen time per store type

train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()
import seaborn as sns

# Compute the correlation matrix 

# exclude 'Open' variable

corr_all = train_store.drop('Open', axis = 1).corr()



 # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_all, square = True, linewidths = .5, cmap = "BuPu")      

plt.show()
# sale per customer trends

sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 

               col = 'Promo',

               hue = 'Promo2',

               palette = 'copper_r') 
train['Sales']
sales_a = train[train.Store == 2]['Sales']

sales_a.describe()

#sales_a.resample('W').sum().plot(color = c)
! ls ../input/rossmann-store-sales
# importing data

df = pd.read_csv("../input/rossmann-store-sales/train.csv")



# remove closed stores and those with no sales

df = df[(df["Open"] != 0) & (df['Sales'] != 0)]



# sales for the store number 1 (StoreType C)

sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]



# reverse to the order: from 2013 to 2015

sales = sales.sort_index(ascending = False)



# to datetime64

sales['Date'] = pd.DatetimeIndex(sales['Date'])

sales.dtypes
# from the prophet documentation every variables should have specific names

sales = sales.rename(columns = {'Date': 'ds',

                                'Sales': 'y'})

sales.head()
# plot daily sales

ax = sales.set_index('ds').plot(figsize = (12, 4), color = c)

ax.set_ylabel('Daily Number of Sales')

ax.set_xlabel('Date')

plt.show()
# set the uncertainty interval to 95% (the Prophet default is 80%)

my_model = Prophet(interval_width = 0.95)

my_model.fit(sales)



# dataframe that extends into future 6 weeks 

future_dates = my_model.make_future_dataframe(periods = 6*7)



print("First week to forecast.")

future_dates.tail(7)
# predictions

forecast = my_model.predict(future_dates)



# preditions for last week

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})
# visualizing predicions

my_model.plot(forecast);
my_model.plot_components(forecast);