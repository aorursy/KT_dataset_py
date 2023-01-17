# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv")

data.info()
data.head()
data.corr()
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns
#line plots

data.Open.plot(kind='line', color='g', label='Open', linewidth=4, alpha=0.5, grid=True, linestyle='-')

data.Close.plot(kind='line', color='r', label='Close', linewidth=2, alpha=1, grid=True, linestyle='--')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Open & Close')

plt.show()
#line plots

data.Open.plot(kind='line', color='g', label='Open', linewidth=4, alpha=1, grid=True, linestyle='-')

data.High.plot(kind='line', color='r', label='High', linewidth=2, alpha=0.5, grid=True, linestyle='--')

plt.legend(loc='upper right')

plt.xlabel('open')

plt.ylabel('high')

plt.title('Open & High')

plt.show()
data.Open.plot(kind='line', color='g', label='Open', linewidth=4, alpha=1, grid=True, linestyle='-')

data.Weighted_Price.plot(kind='line', color='r', label='Weighted_Price', linewidth=2, alpha=1, grid=True, linestyle='--')

plt.legend(loc='upper left')

plt.xlabel('Open')

plt.ylabel('Weighted_Price')

plt.title('Open & Price')

plt.show()