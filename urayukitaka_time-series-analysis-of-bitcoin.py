# Basic library

import numpy as np

import pandas as pd



# Confime the file

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization libraries

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

import cufflinks as cf

import plotly.offline as plyo

plyo.init_notebook_mode(connected=True)





# Time series data preprocessing

import datetime, pytz
raw_data = pd.read_csv("../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", index_col=0, parse_dates=True)

raw_data.head()
# Data size

raw_data.shape
# Data info

raw_data.info()
# Index data

raw_data.index
# Null data

raw_data.isnull().sum()
# Changing to datetime type from timestamp.

data_list = []



for i in raw_data.index:

    k = pytz.utc.localize(datetime.datetime.fromtimestamp(float(i)))

    data_list.append(k)

    

raw_data["datetime"] = data_list
# Fillna by foward data.

raw_data.fillna(method='ffill', inplace=True)
# Checking the dataframe

raw_data.head()
# Timeseries change to per hour.Making the dataframe.

df = pd.DataFrame({})



df["Open"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="H")]).Open.first())["Open"]

df["High"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="H")]).High.max())["High"]

df["Low"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="H")]).Low.min())["Low"]

df["Close"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="H")]).Close.last())["Close"]



df.tail(10)
# data preparation

price = df



# Plot

qf = cf.QuantFig(price, title='Bitcoin', name="Bitcoin")

plyo.iplot(qf.iplot(asFigure=True), image='png', filename='qf_01')
# data preparation, about open price



price_diff = price["Open"].diff()



# Plot, around 2018/Jan

price_diff.iloc[20000:-5000].plot(figsize=(20,6))
# data preparation, about open price



price_change_ratio = price["Open"].pct_change()



# Plot, around 2018/Jan

price_change_ratio.iloc[20000:-5000].plot(figsize=(20,6))
# Timeseries change to per day.Making the dataframe.

df = pd.DataFrame({})



df["Open"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="D")]).Open.first())["Open"]

df["High"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="D")]).High.max())["High"]

df["Low"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="D")]).Low.min())["Low"]

df["Close"] = pd.DataFrame(data=raw_data.groupby([pd.Grouper(key="datetime", freq="D")]).Close.last())["Close"]



df.head(10)
# Set moving average value

df["sma5"] = df["Open"].rolling(window=5).mean()

df["sma25"] = df["Open"].rolling(window=25).mean()

df["sma75"] = df["Open"].rolling(window=75).mean()



# Plot, around 2018/Jan

df.iloc[800:,:][["Open", "sma5", "sma25", "sma75"]].dropna().plot(figsize=(20,6))