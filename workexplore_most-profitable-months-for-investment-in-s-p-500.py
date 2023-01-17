# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



% matplotlib inline



df = pd.DataFrame()



readcols = ['date', 'close']

file = '../input/prices.csv'



"""This program tries to find most profitable months for short term investment in S&P500 . We

draw a chart to display per month percentage variation in All stock price at NYSE over 2012 - 2016."""

for chunk in pd.read_csv(file, chunksize=10000, usecols=readcols, parse_dates=['date']):

	df = df.append(chunk)



# create date index https://stackoverflow.com/questions/35488908/pandas-dataframe-groupby-for-year-month-and-return-with-new-datetimeindex

df = df.set_index('date')



df = df.groupby(['date']).sum()



# resample daily to monthly https://stackoverflow.com/questions/41612641/different-behaviour-with-resample-and-asfreq-in-pandas

df_mon = df.resample('MS').mean()



# remove Nan values

df_mon['delta'] = df_mon.close.pct_change().fillna(0)



#print(df_mon.head(n=50))



# draw 2 plots

f, ax = plt.subplots(2, figsize=(16, 6), sharex=True)



ax[0].plot(df_mon.index, df_mon['close'])

ax[0].set_title("S&P 500 Combined prices - Monthly")

ax[0].set_xlabel("Timeline")

ax[0].set_ylabel("S&P 500 Combined Price in $")



# 

ax[1].plot(df_mon.index, df_mon['delta'])

ax[1].set_title("S&P 500 Combined prices Change Percentage - Monthly")

ax[1].set_xlabel("Timeline")

ax[1].set_ylabel("percentage change")



tmp_index = df_mon.index.strftime("%b")

df_mon = df_mon.set_index(tmp_index)



df_mon.drop(["close"], axis=1, inplace=True)



# group by month name column (index) https://stackoverflow.com/questions/30925079/group-by-index-column-in-pandas

# sort False required....

df_mon_perc = df_mon.groupby([df_mon.index.get_level_values(0)], sort=False).sum()





# label index column as months required to draw ...

df_mon_perc.index.set_names('months', inplace=True)



df_mon_perc['mons'] = df_mon_perc.index



# set string index & plot....https://stackoverflow.com/questions/31523513/plotting-strings-as-axis-in-matplotlib

df_mon_perc = df_mon_perc.set_index('mons')

df_mon_perc.reset_index(drop=True)



plt.show()



# rename column

df_mon_perc.rename(columns={"delta": "Percentage change"}, inplace=True)



# now plot per month chart

df_mon_perc.plot()



plt.xlabel("Months")

plt.title("Percentage Change per month Combined S&P 500 aggregated over 2012 to 2016")

plt.ylabel("Percentage Change from previous month")



plt.show()



# It seems Jun month is most suitable for investment, since it has highest dip

# in percentage compared to previous month.