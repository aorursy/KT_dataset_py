# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
all_data = pd.read_csv("../input/crypto-markets.csv",parse_dates=['date'],index_col=['date'])
all_data.head()
all_data.tail()
all_data['symbol'].unique()
len(all_data['symbol'].unique())
all_data.sort_index().tail(1500).sort_values("ranknow")
btc=all_data[all_data['symbol']=='BTC']
bch = all_data[all_data["symbol"] == 'BCH']
btc = btc.drop(['slug', 'symbol', 'name', 'ranknow','volume', 'market', 'close_ratio', 'spread'],1)
bch = bch.drop(['slug', 'symbol', 'name', 'ranknow','volume', 'market', 'close_ratio', 'spread'],1)
btc.shape
bch.shape # bitcoin cash came out on 1sy aygust 2017
bch.isnull().any()
btc.isnull().any()
btc = btc.sort_index()
new_btc = btc.reset_index()[btc.reset_index()['date'].apply(lambda x: True if x>=pd.to_datetime("2017-08-01") else False)].set_index('date')
new_btc.shape
fig = plt.figure(figsize=(12,6))
plt.figure(figsize=(12,6))
plt.plot(new_btc['close'])
plt.plot(bch['close'])
plt.legend(['BTC','BCH'])

