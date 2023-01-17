# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as dates

import matplotlib as mpl 

import plotly.offline as py

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
rawdata = pd.read_csv('../input/all_stocks_5yr.csv')

dataA = rawdata.loc[rawdata['Name']  == 'T']

dataA.head()
dataA[dataA.open.isnull()]

dataA.dropna(inplace=True)

dataA[dataA.open.isnull()].sum()

data = dataA.set_index('date')

data.head()
fig,ax1 = plt.subplots(figsize=(20, 10))

plt.plot(data[['open','close','high','low']])

plt.show()




trace = go.Candlestick(

                open=data['open'],

                high=data['high'],

                low=data['low'],

                close=data['close'])

d = [trace]

py.iplot(d, filename='simple_candlestick')
data["target"] = (data["close"] - data["close"].shift(-1))/data["close"]

data =data.dropna()

data = data.drop(["Name"],axis=1)

data.head()
data.target.plot()
data_train = data[:-400]

data_test = data[-400:]

test_size = data_test.shape[0]





data_test.shape
go_long_stratgie =  np.ones(test_size)

go_short_stratgie =  -1*np.ones(test_size)

go_hold_stratgie =  np.zeros(test_size)
target = data_test["target"].values
target.shape
(go_long_stratgie*target).sum()
(go_short_stratgie*target).sum()
(go_hold_stratgie*target).sum()
plt.plot(range(test_size),(go_long_stratgie*target).cumsum()) 

plt.show()

def evaluate_and_plot(stratgie,target , show_graph=True):

    print(f"the sum profit is {(stratgie*target).sum()}")

    if show_graph:

        plt.plot(range(test_size),(stratgie*target).cumsum()) 

        plt.show()
evaluate_and_plot(go_long_stratgie,target)
random_monkie01 = np.random.randint(3, size=test_size)-1

random_monkie01[:10]
evaluate_and_plot(random_monkie01,target)
random_monkie1000 = np.random.randint(3, size=(1000,test_size))-1

evaluate_and_plot(random_monkie1000[0],target)

evaluate_and_plot(random_monkie1000[1],target)
max_profit =-20

profits = np.zeros(1000)

for i in range(1000):

    profits[i] = (random_monkie1000[i]*target).sum()
profits[:5]
profits.max()
evaluate_and_plot(random_monkie1000[profits.argmax()],target)
data["bar_change"] = (data["close"] - data["open"])/data["open"]

data["bar_full_size"] = (data["close"] - data["open"])/data["open"]