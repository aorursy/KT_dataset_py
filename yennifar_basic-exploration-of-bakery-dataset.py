# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # data vizualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Dataset overview
df = pd.read_csv('../input/BreadBasket_DMS.csv')
print(df.info())
print(df.head(5))
df[df['Item'] == 'NONE'].count()
df['Year'] = df.Date.apply(lambda x: x.split('-')[0])
df['Month'] = df.Date.apply(lambda x: x.split('-')[1])
df['Day'] = df.Date.apply(lambda x: x.split('-')[2])
df['Hour'] = df.Time.apply(lambda x: int(x.split(':')[0]))
df.drop(columns = 'Time', inplace = True)
df = df[df['Item'] != 'NONE']
unique_items = len(df['Item'].unique())
print('Unique items sold: ' + str(unique_items))
sns.set(style = 'whitegrid')
sales = df['Item'].value_counts()
f = sales[:10].plot.bar(title = 'Top 10 sales')
f.legend(['Number of items sold'])
coffee_sales = df[df['Item'] == 'Coffee']
coffee_times = coffee_sales['Hour'].value_counts().sort_index()
f = coffee_times.plot.bar(title = 'Coffee sales by hour')
f.set_xlabel('Time of day')
frequent_items = sales[1:10] #skip coffee
for item in frequent_items.index:
    plt.figure()
    curr_sales = df[df['Item'] == item]
    curr_times = curr_sales['Hour'].value_counts().sort_index()
    f = curr_times.plot.bar(title = (item + ' sales by hour'))
    f.set_xlabel('Time of day')
df['Day_of_week'] = pd.to_datetime(df['Date']).dt.weekday_name
sales_by_day = df['Day_of_week'].value_counts()
sales_by_day.plot.bar(title = 'Sales by day of week')
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_coffee = df[df['Item'] == 'Coffee']
for day in weekdays:
    plt.figure()
    curr_sales = df_coffee[df_coffee['Day_of_week'] == day]
    curr_times = curr_sales['Hour'].value_counts().sort_index()
    curr_times.plot.bar(title = (day + ' coffee sales by hour'))
transactions_by_month = pd.DataFrame(df.groupby(by = ['Year', 'Month'])['Transaction'].nunique().rename('N transactions')).reset_index()
transactions_by_month['Date'] = transactions_by_month['Year'] + '-' + transactions_by_month['Month']
g = sns.barplot('Date', 'N transactions', data = transactions_by_month)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)
transactions_by_date = pd.DataFrame(df.groupby(by = ['Year', 'Month', 'Day'])['Transaction'].nunique().rename('Transactions a day')).reset_index()
transactions_by_date['Date'] = transactions_by_date['Year'] + '-' + transactions_by_date['Month']
g = sns.barplot('Date', 'Transactions a day', data = transactions_by_date)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)
df_holidays = df[df['Month'] == '12']
df_holidays = df_holidays[df_holidays['Day'].isin(map(str, range(24, 32)))]
print(df_holidays.shape)
holiday_by_date = pd.DataFrame(df_holidays.groupby(by = ['Month', 'Day'])['Transaction'].nunique().rename('Transactions a day')).reset_index()
holiday_by_date['Date'] = holiday_by_date['Month'] + '-' + holiday_by_date['Day']
g = sns.barplot('Date', 'Transactions a day', data = holiday_by_date)
g.set_xticklabels(g.get_xticklabels(), rotation = 30)