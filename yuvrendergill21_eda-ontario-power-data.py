import pandas as pd

import numpy as np

import matplotlib

%matplotlib inline

import os

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as offline

import math



import plotly.plotly as py



offline.init_notebook_mode(connected=True)



import plotly.graph_objs as go

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf,pacf

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from keras.layers import Dense, LSTM, Dropout

from keras.models import Sequential

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



from geopy.geocoders import Nominatim

import gc

import time

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
power = pd.read_csv('../input/Master.csv' ,

                    parse_dates=['Date'])



df = power

df['DateTime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')

df['TimeStamp'] = df.DateTime.values.astype(np.int64) // 10**9

df.drop(['Date','Hour','TimeStamp'],axis=1,inplace=True)

df.set_index('DateTime', inplace=True, drop=True)

df.fillna(0,inplace=True)

df.head()
sns.distplot(df['Ontario Demand'], color="r");



num_df = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 0:21]

m = len(num_df.columns)



fig = plt.figure(dpi = 200, figsize = (30, 25))



for i in range(1, m):

    col = num_df.iloc[:, i]

    ax = fig.add_subplot(math.ceil(m / 4), 4, i)

    ax.plot = sns.distplot(col[~ np.isnan(col)], hist = True, kde = True,

                 kde_kws = {'shade': False, 'linewidth': 1}, color='g')

    plt.xlabel(num_df.columns[i], fontsize = 10)

    plt.ylabel("Probability")


df['Ontario Demand'].plot(figsize = (25,7))


num_df = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 0:21]

m = len(num_df.columns)



fig = plt.figure(dpi = 200, figsize = (30, 25))



for i in range(1, m):

    col = num_df.iloc[:, i]

    ax = fig.add_subplot(math.ceil(m / 4), 4, i)

    ax.plot = col[~ np.isnan(col)].plot()

    plt.xlabel(num_df.columns[i], fontsize = 10)

    plt.ylabel("Power Demand (MW)")
def get_category(txt, bag):

    category = [ 1 for x in bag if x == txt]

    if not category:

        category = [0]

    return category[0] 
# Adding extra features for more indepth analysis 

df['date'] = df.index

df['hour'] = df['date'].dt.hour

df['dayofweek'] = df['date'].dt.dayofweek

df['quarter'] = df['date'].dt.quarter

df['month'] = df['date'].dt.month

df['year'] = df['date'].dt.year

df['dayofyear'] = df['date'].dt.dayofyear

df['dayofmonth'] = df['date'].dt.day

df['weekofyear'] = df['date'].dt.weekofyear

summer_months = [5, 6, 7, 8]

df['summer'] = df['month'].apply(lambda x : get_category(x, summer_months))

df.tail()
def plot_monthly_avg(df, zone, start_year, end_year):

    df = df.loc[(start_year <= df.year) & (df.year <= end_year)]

    temp = df[[zone,'date']]

    temp.loc[:,'date'] = pd.to_datetime(temp['date'])

    temp.loc[:,'Month'] = temp['date'].dt.month

    count_mean = temp.groupby('Month')[zone].agg(['count','mean', 'max', 'min']).sort_index()

    count_mean.rename(columns={'count':'Counts','mean':'Average',  'max': 'Peak', 'min':'Min'},inplace=True)

    fig = plt.figure(dpi = 200, figsize = (20, 15))

    for idx,col in enumerate(count_mean.columns):

        if idx > 0:

            ax = fig.add_subplot(4, 2, idx)

            ax.plot = sns.barplot(x=count_mean.index,y=col,data=count_mean,ax=ax)

            ax.set_title("Distribution of Power Demand {} Among Months".format(col),fontsize=20)

            plt.ylabel("Power Dmeand (MW)")

            plt.xlabel("")



def plot_weekly_avg(df, zone, start_year, end_year):

    df = df.loc[(start_year <= df.year) & (df.year <= end_year)]

    temp = df[[zone,'date']]

    temp.loc[:,'date'] = pd.to_datetime(temp['date'])

    temp.loc[:,'Month'] = temp['date'].dt.weekday

    count_mean = temp.groupby('Month')[zone].agg(['count','mean', 'max', 'min']).sort_index()

    count_mean.rename(columns={'count':'Counts','mean':'Average', 'max': 'Peak', 'min':'Min'})

    count_mean.rename(index={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'},inplace=True)

    fig = plt.figure(dpi = 200, figsize = (20, 15))

    for idx,col in enumerate(count_mean.columns):

        if idx > 0:  

            ax = fig.add_subplot(4, 2, idx)

            sns.barplot(count_mean.index,count_mean[col],ax=ax)

            ax.set_title("Distribution of Power Demand {} Among Weekdays".format(col),fontsize=16)

            plt.ylabel("Power Demand (MW)")

            plt.xlabel("")

            

plot_monthly_avg(df, 'Ontario Demand', 2018, 2018)            

plot_weekly_avg(df, 'Ontario Demand', 2018, 2018)