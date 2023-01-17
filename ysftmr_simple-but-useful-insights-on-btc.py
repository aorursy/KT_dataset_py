# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
btc=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv")  #importing csv file 
btc.info()        #quick view on the data to understand what the data is about
btc.columns    #quick view on the data to understand what the data is about
btc.head()    #quick view on the data to understand what the data is about
btc["Timestamp"]=pd.to_datetime(btc["Timestamp"],unit="s")       #convert unix timestamps to human date and time to make 
                                                                 #it comprehensible easily.
btc.fillna(0,inplace=True)   # replace all "NaN" values with 0 to make it seem more simple
                             # use inplace=True parameter to make permanent change on the data
btc.tail()     #quick look after the changes made
maximum=-1                          # find maximum and minimum values in the data 
for each in btc.Weighted_Price:     # of course you can simply use <btc.Weighted_Price.max()> or <btc.Weighted_Price.min()>
    if each>maximum:                # my purpose is to use for loop and if conditon to reiterate
        maximum=each                # just like when you are learning a new language, using the new words in the sentence to 
print(maximum)                      # make practice

minimum=100000
for each in btc.Weighted_Price:
    if each<minimum:
        minimum=each    
print(minimum)
day=btc["Timestamp"]==btc["Timestamp"].dt.floor("D")  # there are over 3 million entries in the dataframe
btc_daily=btc[day]                                    # to make the dataset more simple i only take daily values
btc_daily.info()                                      # so that the entry number dropped to 2502
plt.plot(btc_daily.Timestamp, btc_daily.Weighted_Price, color="green", label="BTC/USD")      # line plot for seeing the daily weighted price
plt.rcParams["figure.figsize"]=(40,15)
plt.xlabel ("Time")
plt.ylabel("USD")
plt.legend() 
plt.show()
btc_daily.corr()      # correlation between columns, the closer to "1", the more correlated 
f,ax = plt.subplots(figsize=(30, 10))                                        # as we can see from the heatmap, exluding Volume_Btc and Volume_currency, other columns
sns.heatmap(btc_daily.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)   # strongly correlated as expected.
plt.show()
