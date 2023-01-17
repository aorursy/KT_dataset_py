# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv') 

# İnitializing the csv file.

data.info() # Get to know my dataset headers.

data.corr() # Putting my variables into table.
data.head() # Need to fix NaN values.
data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(value=0,inplace=True) # Filled na's with zeros
# I will do the same to (open,high,low,close), these are continious series so i will use

#'ffill'

data['Open'].fillna(method='ffill', inplace=True)

data['High'].fillna(method='ffill', inplace=True)

data['Low'].fillna(method='ffill', inplace=True)

data['Close'].fillna(method='ffill', inplace=True)
data.head() # As you can see NaN's are disappeared from Both(Volume_(BTC)...to, Open...)
data.tail() # Its forwarding that means its the right solution.
f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(data.corr(), annot = True, linewidths = .5,fmt='.1f', ax=ax )

plt.show  # Visualizated our dataset with our one method.
data.columns
fig, ax = plt.subplots(figsize=(20,10))

data["Volume_(BTC)"].plot(kind='line', color='b', label='Volume_(BTC)',

                          linewidth = 3, alpha=0.5, grid=True, linestyle= ':')



data["Weighted_Price"].plot(kind='line', color='g', label='Weighted_Price',

                            linewidth = 3, alpha=0.5, grid=True, linestyle= '-.')



plt.legend(loc='upper left')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show() # Graphic of Volume/Price 
data["Volume_(BTC)"].plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()