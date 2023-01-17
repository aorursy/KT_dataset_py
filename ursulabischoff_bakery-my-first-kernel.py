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
df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head()
df.info()
df['datetime'] = pd.to_datetime(df['Date']+' '+ df['Time'])
df['day_of_week'] = df['datetime'].dt.day_name()
total_items = df['Transaction'].count()
unique_transactions = df['Transaction'].nunique()

items_per_transaction = total_items / unique_transactions
print(total_items, unique_transactions,items_per_transaction)
new_df = pd.pivot_table(df, values='Transaction', columns='day_of_week', aggfunc=('count','nunique')).transpose().reset_index()
new_df = new_df.sort_values(by='count', ascending=False)
total_items_by_weekday = df['day_of_week'].value_counts()
print(total_items_by_weekday.sort_values(ascending=False))
unique_transactions_by_weekday = df.groupby('day_of_week')['Transaction'].nunique()
print(unique_transactions_by_weekday.sort_values(ascending=False))
items_per_transaction_by_weekday = total_items_by_weekday / unique_transactions_by_weekday
print(items_per_transaction_by_weekday.sort_values(ascending=False))
new_df.plot.bar('day_of_week')
df['Item'].value_counts().head(20)
df['Item'].nunique()
list_of_items = df['Item'].unique().tolist()
print(list_of_items)
df.set_index('datetime', inplace=True, drop=True)
df_ts = df.drop(['Date', 'Time', 'day_of_week', 'Item'], axis=1)
weekly = df_ts.resample('W-MON').agg(['count', 'nunique'])
weekly['Transaction'].plot(kind='line')
monthly = df_ts.resample('M').agg(['count', 'nunique'])
monthly['Transaction'].plot(kind='line')