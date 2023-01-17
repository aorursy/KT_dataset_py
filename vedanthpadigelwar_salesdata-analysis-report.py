# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns 



import warnings

# current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')



import missingno as msno # missing data visualization module for Python

import pandas_profiling



import gc

import datetime



%matplotlib inline

color = sns.color_palette()
sales_data=pd.read_csv("../input/sales-data-postman/sales_data.csv")  

date=pd.read_csv("../input/sales-data-postman/date.csv")  
sales_data.head()
sales_data.tail()
date.head()
sales_data.describe()

print(sales_data.info())

print(date.info())
# renaming column names

sales_data.rename(index=str, columns={'transaction id': 'transaction_id',

                              'product id' : 'product_id',

                              'product description' : 'product_description',

                              'quantity sold' : 'quantity_sold',

                              'transaction timestamp':'timestamp',

                              'unit price' : 'unit_price',

                              'customer id' : 'customer_id',

                              'transaction country' : 'transaction_country'}, inplace=True)

date.rename(index=str, columns={'timestamp              ':'timestamp'}, inplace=True)
#changing values of timestamp in  datetime64 format

sales_data['timestamp'] = pd.to_datetime(sales_data.timestamp, format='%d/%m/%Y %H:%M')

date['timestamp'] = pd.to_datetime(date.timestamp, format='%d/%m/%Y %H:%M')
# getting values hour, day, month, year_month from  timestamp to get sales analysis and adding to main dataframe

sales_data.insert(loc=4, column='day', value=(sales_data.timestamp.dt.dayofweek)+1)

sales_data.insert(loc=5, column='hour', value=sales_data.timestamp.dt.hour)

sales_data.insert(loc=3, column='month', value=sales_data.timestamp.dt.month)

sales_data.insert(loc=2, column='year_month', value=sales_data['timestamp'].map(lambda x: 100*x.year + x.month))



# Adding total price per product in a order

sales_data['Total_price_per_product'] = sales_data['quantity_sold'] * sales_data['unit_price']

sales_data
# try and tested rough code



# chaning index values 

#sales_data.set_index("timestamp")

#date.set_index("timestamp")



# printing shapes of both the data

#print(sales_data.shape)

#print(date.shape)



# combining both the data using merge

#sales_data.merge(date, on="timestamp", how='inner')



# combining both the dataframes using concat

#sales=pd.concat([sales_data,date],axis=1,join='inner')

#sales
#checking null values

print(sales_data.isnull().sum())

# displaying null values

sales_data[sales_data.isnull().any(axis=1)]
# dropping null values

final_sales = sales_data.dropna()

final_sales.isnull().sum()
# a general description of the data

final_sales.describe().round(2)
#Remove Quantity with negative values

final_sales = final_sales[final_sales.quantity_sold > 0]

final_sales.describe()
# displaying sales data

print(final_sales.info())

final_sales.head()

#changing customer id to int  

final_sales['customer_id'] = final_sales['customer_id'].astype('int64')
final_sales.describe()
print(final_sales.groupby(by=['customer_id','transaction_country'], as_index=False)['transaction_id'].count())
orders = final_sales.groupby(by=['customer_id','transaction_country'], as_index=False)['transaction_id'].count()



plt.subplots(figsize=(15,6))

plt.plot(orders.customer_id, orders.transaction_id)

plt.xlabel('Customers ID')

plt.ylabel('Number of Orders')

plt.title('Number of Orders for different Customers')

plt.show()
print('The TOP 5 customers with most number of orders...')

orders.sort_values(by='transaction_id', ascending=False).head()
final_sales.groupby(by=['customer_id','transaction_country'], as_index=False)['Total_price_per_product'].sum()
money_spent = final_sales.groupby(by=['customer_id','transaction_country'], as_index=False)['Total_price_per_product'].sum()



plt.subplots(figsize=(15,6))

plt.plot(money_spent.customer_id, money_spent.Total_price_per_product)

plt.xlabel('Customers ID')

plt.ylabel('Money spent (Dollar)')

plt.title('Money Spent by different Customers')

plt.show()
print('The TOP 5 customers with highest money spent...')

money_spent.sort_values(by='Total_price_per_product', ascending=False).head()
final_sales.groupby('year_month')['Total_price_per_product'].sum()
ax=final_sales.groupby('year_month')['Total_price_per_product'].sum().plot(kind='bar',color='b',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Amount earned per month($ in million)',fontsize=15)

ax.set_title('Revenue for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()
amt_per_month=final_sales.groupby('year_month')['Total_price_per_product'].sum()

count_per_month=final_sales.groupby('transaction_id')['year_month'].unique().value_counts().sort_index()

ax= amt_per_month.div(count_per_month).plot(kind='bar',color='b',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Average order value per month($ )',fontsize=15)

ax.set_title('Average order value for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()
temp=(final_sales.groupby(by=['year_month'])['Total_price_per_product'].mean()).plot(kind='bar',color='b',figsize=(15,6))

ax=temp

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Average product value',fontsize=15)

ax.set_title('Average product price for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()
final_sales.head()
ax = final_sales.groupby('transaction_id')['year_month'].unique().value_counts().sort_index().plot(kind='bar',color='b',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()
final_sales.groupby('transaction_id')['day'].unique().value_counts().sort_index()
ax = final_sales.groupby('transaction_id')['day'].unique().value_counts().sort_index().plot(kind='bar',color='b',figsize=(15,6))

ax.set_xlabel('Day',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Days',fontsize=15)

ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'), rotation='horizontal', fontsize=15)

plt.show()
final_sales.groupby('transaction_id')['hour'].unique().value_counts().iloc[:-1].sort_index()
ax = final_sales.groupby('transaction_id')['hour'].unique().value_counts().iloc[:-1].sort_index().plot(kind='bar',color='b',figsize=(15,6))

ax.set_xlabel('Hour',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Hours',fontsize=15)

ax.set_xticklabels(range(6,21), rotation='horizontal', fontsize=15)

plt.show()
final_sales.unit_price.describe()
# check the distribution of unit price

plt.subplots(figsize=(12,6))

sns.boxplot(final_sales.unit_price)

plt.show()
df_free = final_sales[final_sales.unit_price == 0]

df_free.head()
df_free.year_month.value_counts().sort_index()
ax = df_free.year_month.value_counts().sort_index().plot(kind='bar',figsize=(12,6), color='b')

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Frequency',fontsize=15)

ax.set_title('Free sale Frequency for different Months (Dec 2010 - Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11'), rotation='horizontal', fontsize=13)

plt.show()
final_sales.head()
group_country_orders = final_sales.groupby('transaction_country')['transaction_id'].count().sort_values()

# del group_country_orders['United Kingdom']



# plot number of unique customers in each country (with UK)

plt.subplots(figsize=(15,8))

group_country_orders.plot(kind='bar', fontsize=12, color='b')

plt.xlabel('Number of Orders', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title('Number of Orders for different Countries', fontsize=12)

plt.show()
group_country_amount_spent = final_sales.groupby('transaction_country')['Total_price_per_product'].sum().sort_values()

# del group_country_orders['United Kingdom']



# plot total money spent by each country (with UK)

plt.subplots(figsize=(15,8))

group_country_amount_spent.plot(kind='bar', fontsize=12, color='b')

plt.xlabel('Money Spent (Dollar)', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title('Money Spent by different Countries', fontsize=12)

plt.show()
group_country_amount_spent = final_sales.groupby('transaction_country')['Total_price_per_product'].sum().sort_values()

del group_country_amount_spent['United Kingdom']



# plot total money spent by each country (without UK)

plt.subplots(figsize=(15,8))

group_country_amount_spent.plot(kind='bar', fontsize=12, color='b')

plt.xlabel('Money Spent (Dollar)', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title('Money Spent by different Countries', fontsize=12)

plt.show()