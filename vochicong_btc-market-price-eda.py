# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# os.listdir("../input/bitcoin-historical-data")
def read_csv(path):
    data = pd.read_csv(
        path,
#         index_col=0,
#         parse_dates=True,
#         parse_dates={'date': [0]},
#         date_parser=data.Timestamp,
        usecols=['Timestamp', 'Volume_(BTC)', 'Weighted_Price'],
#         skiprows=(lambda t: t == 3)
    )
    data['datetime'] = pd.to_datetime(data.Timestamp, unit='s')
#     start = '2012-01-01 08:00:00'
    start = '2017-07-04 08:00:00'
#     start = '2017-11-08 09:13:00'
#     start = '2018-06-26 20:00:00'
    data = data[data.datetime > start ]
    data.set_index('datetime', inplace=True)
    data.rename(index=str, columns={"Volume_(BTC)": "volume", 'Weighted_Price': "price"}, inplace=True)
    return data

bitflyerJPY = read_csv('../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv')
bitflyerJPY.head()
bitflyerJPY.tail()
coinbaseUSD = read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv')
coinbaseUSD.head()
coinbaseUSD.tail()
coincheckJPY = read_csv('../input/coincheckJPY_1-min_data_2014-10-31_to_2018-06-27.csv')
coincheckJPY.head()
bitstampUSD = read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
bitstampUSD.head()
for data in [bitflyerJPY, coincheckJPY, coinbaseUSD, bitstampUSD]:
    print(len(data))
    print(data.isnull().values.any())
# list(map(lambda t: len(t), [pd.to_datetime(np.setdiff1d(bitflyerJPY.index, coincheckJPY.index), unit='s'),
# pd.to_datetime(np.setdiff1d(coincheckJPY.index, coinbaseUSD.index), unit='s'),
# pd.to_datetime(np.setdiff1d(coinbaseUSD.index, bitstampUSD.index), unit='s'),
# pd.to_datetime(np.setdiff1d(bitstampUSD.index, bitflyerJPY.index), unit='s')]
# ))
bitflyerJPY.price.plot.line()
coincheckJPY.price.plot.line()
# pd.DataFrame([bitflyerJPY.price, coincheckJPY.price]).T
dfJPY = pd.DataFrame({'bitflyerJPY': bitflyerJPY.price, 'coincheckJPY': coincheckJPY.price})
dfJPY
dfJPY.plot.line()
coinbaseUSD.price.plot.line()
bitstampUSD.price.plot.line()
pd.DataFrame({'coinbaseUSD': coinbaseUSD.price, 'bitstampUSD': bitstampUSD.price}).plot.line()


dfall = pd.DataFrame({'bitflyerJPY': bitflyerJPY.price, 'coincheckJPY': coincheckJPY.price,
             'coinbaseUSD': coinbaseUSD.price, 'bitstampUSD': bitstampUSD.price})
dfall.plot.line()
# from sklearn.preprocessing import normalize
dfall.head()
dfall.iloc[0]
(dfall / dfall.iloc[0]).plot.line()
start = -100
def plot(df, start=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if start:
        df = df.iloc[start:]/df.iloc[start] 
    ax = df.plot.line(figsize=(12, 6),
    #     color='mediumvioletred',
    #     fontsize=16,
        title='BTC price',)
    ax.set_title("BTC price", fontsize=20)
    sns.despine(bottom=True, left=True)
plot(dfall, -60*24*30) # last month
plot(dfall, -60*24) # last day
plot(dfall, -60) # last hour
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df/df.shift(1)) - 1
    daily_returns.iloc[0,:] = 0
    return daily_returns


def plot_daily_returns(df, title='BTC returns (%)'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = compute_daily_returns(df) * 100
    ax = df.plot.line(figsize=(12, 6),
                     )
    ax.set_title(title, fontsize=20)
#     sns.despine(bottom=True, left=True)


plot_daily_returns(dfall[-60*1:], 'BTC minutely returns (%)')
plot_daily_returns(dfall[-60:-30], 'BTC minutely returns (%)')
len(dfall)/(60*24)
dfall.head()
dfall.describe()
dfall.index = pd.to_datetime(dfall.index)
dfall.resample('m').mean()
dfall.resample('d').mean()
dfall.resample('h').mean()
plot_daily_returns(dfall.resample('m').mean(), 'BTC monthly returns (%)')
plot_daily_returns((dfall.resample('d').mean())[-30:], 'BTC daily returns (%)')
plot_daily_returns((dfall.resample('h').mean())[-24:], 'BTC hourly returns (%)')
plot_daily_returns(dfall[-60*1:], 'BTC minutely returns (%)')