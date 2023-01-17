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
datasource = 'https://raw.githubusercontent.com/phamdinhkhanh/AISchool/master/data_stocks.csv?fbclid=IwAR3FFvX0KkhZsJ1S35xG8ogI225A_3t1LqXK9WrsDwmYTVa-KGKGVXqYnQc'

dataset = pd.read_csv(datasource, header=0, index_col=0)



print(dataset.iloc[:5, :])



# print(dataset.shape)

# print(dataset.describe())



# print(dataset.melt('Symbols').head())



# 1. Tính mức giá Open, High, Low, Close trung bình của mỗi mã chứng khoán trong thời gian tồn tại.



# sorted_mean_symbols = dataset.groupby(['Symbols'], as_index=False)\

#                         .mean()\

#                         .sort_values(by=['Symbols'])\



# print(sorted_mean_symbols)



sorted_mean_symbols = pd.pivot_table(dataset,

                                     values = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],

                                     index = ['Symbols'],

                                     aggfunc = np.mean)

print(sorted_mean_symbols)
date_list = dataset.Date.unique()

symbols_list = dataset.Symbols.unique()

symbols_dates_datasource = []



for date in date_list:

    symbol_close_by_date_list = [date]

    for symbol in symbols_list:

        symbol_close_by_date = dataset.loc[(dataset['Date'] == date) & (dataset['Symbols'] == symbol)]['Close'].values

        if len(symbol_close_by_date) :

            symbol_close_by_date_list.append(symbol_close_by_date[0])

        else:

            symbol_close_by_date_list.append(np.nan)

    symbols_dates_datasource.append(symbol_close_by_date_list)



symbols_dates_dataset = pd.DataFrame(symbols_dates_datasource, columns=np.append(["Date"], symbols_list)).fillna(0)



stock = pd.pivot_table(symbols_dates_dataset, values=symbols_list, index=["Date"])

# print(stock)



stock_change = stock.apply(lambda x: np.log(x) - np.log(x.shift(1)))



# print(stock_change)

print(stock_change.replace([np.inf, -np.inf], np.nan).dropna())

# stock_change.plot()

stock_change.replace([np.inf, -np.inf], np.nan).dropna().plot()
# stock_change.describe()

stock_change.replace([np.inf, -np.inf], np.nan).dropna().describe()