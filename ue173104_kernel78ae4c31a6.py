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
from kaggle.competitions import twosigmanews
env=twosigmanews.make_env()

(market_train_df,news_train_df)=env.get_training_data()
m_dim=market_train_df.shape
print(f'Market dataset has  {m_dim[0]}  sample and {m_dim[1]} feature ')
market_train_df.head()
market_train_df.dtypes
news_train_df.head()
n_dim=news_train_df.shape
print(f'News dataset has  {n_dim[0]}  sample and {n_dim[1]} feature ')
news_train_df.dtypes
news_train_df['assetName'].unique()
news_train_df['assetName'].value_counts()
market_train_df['assetName'].unique()
market_train_df['assetName'].value_counts()
market_train_df.dtypes
market_train_df.isnull().sum()*100/market_train_df.shape[0]
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(15,10))
market_train_df['returnsClosePrevMktres1'].plot(kind='box')
( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].mean()).value_counts()
( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].median()).value_counts()
market_train_df['returnsClosePrevMktres1'].fillna(market_train_df['returnsClosePrevMktres1'].mean(),inplace=True)
market_train_df['returnsOpenPrevMktres1'].plot.box()
market_train_df['returnsOpenPrevMktres1'].fillna(market_train_df['returnsOpenPrevMktres1'].median(),inplace=True)
plt.figure(figsize=(15,10))
market_train_df['returnsClosePrevMktres10'].plot.box()
market_train_df['returnsClosePrevMktres10'].fillna(market_train_df['returnsClosePrevMktres10'].median(),inplace=True)
plt.figure(figsize=(15,10))
market_train_df['returnsOpenPrevMktres1'].plot.box()
market_train_df['returnsOpenPrevMktres10'].fillna(market_train_df['returnsOpenPrevMktres10'].median(),inplace=True)
market_train_df.isnull().sum()
plt.figure(figsize=(14,7))
plt.plot(market_train_df['time'],market_train_df['volume'],label='Time vs Volume')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()


