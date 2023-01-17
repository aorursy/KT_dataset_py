# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.use_inf_as_na = True

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats.mstats import gmean

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
price_df = pd.read_csv('../input/nyse/prices.csv')

sec_df = pd.read_csv('../input/nyse/securities.csv')

fund_df = pd.read_csv('../input/nyse/fundamentals.csv')
price_df.head()
price_df.isna().sum()
sec_df = sec_df.rename(columns = {'Ticker symbol' : 'symbol','GICS Sector' : 'sector'})

price_df  = price_df.merge(sec_df[['symbol','sector']], on = 'symbol')

price_df['date'] = pd.to_datetime(price_df['date'])

price_df['year'] = price_df['date'].map(lambda x: x.year)

#return here is log-returns

price_df['return'] = np.log(price_df.close / price_df.close.shift(1)) + 1

#delete entries at the start of each stock

price_df['return_valid'] = price_df['symbol'] == price_df['symbol'].shift(1)

price_df = price_df.drop(price_df[price_df['return_valid'] == False].index)

price_df.dropna(how='any', thresh=None, subset=None, inplace = True)



price_df.tail()
sns.distplot(price_df['return'])

price_df['return'].describe()
std = price_df['return'].std()

mean = price_df['return'].mean()

low = mean-std*5

high = mean+std * 5

fig, ax = plt.subplots(figsize = [25,5])

sns.distplot(np.clip(price_df['return'],low,high), ax = ax)
fund_df = fund_df.rename(columns = {'Ticker Symbol' : 'symbol', 'For Year':'year', 'Period Ending': 'date'})

fund_df  = fund_df.merge(sec_df[['symbol','sector']], on = 'symbol')

fund_df
risk_free_rate = {2011:0.0198,

                  2012:0.0178, 

                  2013:0.0304, 

                  2014:0.0217, 

                  2015:0.0227, 

                  2016:0.0244,

                  2017:0.0241}
def f(row):

    next_year_data = price_df[(price_df['symbol'] == row['symbol']) & (price_df['year'] == row['year']+1)]['return']

    current_data = price_df[(price_df['symbol'] == row['symbol']) & (price_df['year'] == row['year'])]

    #print(data)

    try: 

        calculateRatios(row,next_year_data,current_data['close'],risk_free_rate[row['year']+1])  

    finally:

        return row



def calculateRatios(row,data,pricedata,rf): 

    row['closing_price'] = pricedata.iloc[-1]

    row['return'] = (np.mean(data) - 1)*len(data)

    row['stdev'] = data.std()*np.sqrt(len(data))

    row['sharpe'] = (row['return'] - rf) / row['stdev']
#check that our code outputs correct result



row = 100

f_sharpe = fund_df[row:row+1].apply(f, axis = 1)['sharpe']

print(fund_df[row:row+1])



data = price_df[(price_df['symbol'] == 'AME') & (price_df['year'] == 2013)]['return']

returns = (np.mean(data) - 1)*len(data)

print(returns)

stdev = data.std()*np.sqrt(len(data))

print(stdev)

sharpe = (returns - risk_free_rate[2013]) / stdev



print("Manually Computed: ", sharpe, " Function Value: ", float(f_sharpe))
fund_df = fund_df.apply(f, axis = 1)

fund_df
fund_df = fund_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

sns.countplot(fund_df['year'])
sns.distplot(fund_df['sharpe'])

fund_df['sharpe'].describe()
len(fund_df[fund_df['sharpe'] > 1])/len(fund_df)
fund_df.loc[:,'P/E'] = fund_df['closing_price']/fund_df['Earnings Per Share']

fund_df.loc[:,'TEV/NI'] = fund_df['Total Liabilities & Equity']/fund_df['Net Income']

fund_df.drop('closing_price', axis = 1)
price_df.isna().sum()
fund_df.isna().sum().sum()
sec_df.isna().sum()
fund_df.to_csv("fund.csv", index=False)

sec_df.to_csv("sec.csv", index=False)

price_df.to_csv("price.csv", index=False)