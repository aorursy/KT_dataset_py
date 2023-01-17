# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Load libraries

import matplotlib.pyplot as plt

import traceback

% matplotlib inline



readcols = ['Date', 'Close']

df = pd.read_csv("../input/bitcoin_price.csv", usecols=readcols, parse_dates=['Date'])



# create date index https://stackoverflow.com/questions/35488908/pandas-dataframe-groupby-for-year-month-and-return-with-new-datetimeindex

df = df.set_index('Date')



#print(df.index)



#plt.figure()

f, ax = plt.subplots(2, sharex=True, figsize=(16, 6))



ax[0].plot(df)

ax[0].set_title("Bitcoin historic prices - Daily")

ax[0].set_xlabel("Timeline")

ax[0].set_ylabel("Bitcoin Price in $")



#plt.show()



# resample daily to monthly https://stackoverflow.com/questions/41612641/different-behaviour-with-resample-and-asfreq-in-pandas

df_mon = df.resample('M').mean()



#print(df_mon)



ax[1].plot(df_mon)

ax[1].set_title("Bitcoin Average prices - Monthly")

ax[1].set_xlabel("Timeline")

ax[0].set_ylabel("Bitcoin Price in $")



plt.show()
