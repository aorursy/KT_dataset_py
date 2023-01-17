import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import glob
path = '../input/monthly-sales-2019'

all_files = glob.glob(path + "/*.csv")

data = pd.DataFrame()

for file in all_files:

    df = pd.read_csv(file)

    data = pd.concat([data, df])

    

# data.to_csv('All_Months_Data.csv', index=False)
data.head()
data.sort_values(by=['Order ID'], inplace=True) # Sorting by Order ID
data.tail() # looking at the last values in a dataset
data.isna().sum()
nan_df = data[data.isna().any(axis=1)] 

nan_df.head()
data.dropna(how='all', inplace=True)

data.isna().sum()
data.info()
data.columns = ['Order_ID', 'Product', 'Quantity_Ordered', 'Price_Each', 'Order_Date', 'Purchase_Address']

data.head()
data['Date'] = data['Order_Date'].str[:8]

data['Month'] = data['Order_Date'].str[:2]

data.head()
data.Month.value_counts()
temp_data = data[data['Order_Date'].str[:2] == 'Or']

temp_data
data = data[data['Order_Date'].str[:2] != 'Or']

data.head()
data.shape  # there is 185950 rows in a dataset
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data['Date'] = pd.to_datetime(data['Date'])

data['Month'] = data['Month'].astype('int32')

data.info()
data.Order_ID.value_counts()
data.Quantity_Ordered.value_counts()
data['Quantity_Ordered'] = data['Quantity_Ordered'].astype('int32')

data['Price_Each'] = data['Price_Each'].astype('float64')

data.info()
data.describe()
data['Sales'] = data['Quantity_Ordered'] * data['Price_Each']

data.head()
data.groupby('Month').sum()
result_month = data.groupby(data.Date.dt.month).sum()

result_month
result_year = data.groupby(data.Date.dt.year).sum()

result_year
months = range(1, 13)

plt.bar(months, result_month['Sales'])

plt.title('Sales per Month')

plt.xticks(months)

plt.xlabel('Months')

plt.ylabel('Sales')

plt.show()
data['City'] = data['Purchase_Address'].apply(lambda x: x.split(',')[1])

data.head()
data.City.value_counts()
state = data['Purchase_Address'].apply(lambda x: x.split(', ')[2])

state = state.str[:2]

data['State'] = state

data.head()
result_city = data.groupby(['City', 'State']).sum()

result_city
result_city.index
data['Address'] = data['City'] + " " + data['State']

data.head()
result_address = data.groupby(['Address']).sum()

result_address
city = [c for c, df in data.groupby('Address')]

plt.bar(city, result_address['Sales'])

plt.title('Sales per City')

plt.xticks(result_address.index, rotation='vertical')

plt.xlabel('Cities')

plt.ylabel('Sales')

plt.show()
data.groupby('Product')['Quantity_Ordered'].count()
data['Hour'] = data.Order_Date.dt.hour

data.head()
data.groupby('Hour').count()
hours = [hour for hour, df in data.groupby('Hour')]

plt.plot(hours, data.groupby('Hour').count())

plt.grid(True)

plt.xticks(hours)

plt.xlabel('Hours in 24 Hr Format')

plt.ylabel('Orders')

plt.show()
data.head()
data[data['Order_ID'].duplicated(keep=False)]
df = data[data['Order_ID'].duplicated(keep=False)]

df['Grouped'] = df.groupby('Order_ID')['Product'].transform(lambda x: ','.join(x))

df = df[['Order_ID', 'Grouped']].drop_duplicates()

df.head()
from itertools import combinations

from collections import Counter



count = Counter()



for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 2)))

    

for key, value in count.most_common(10):

    print(key, value)
count = Counter()

for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 3)))

    

for key, value in count.most_common(10):

    print(key, value)
count = Counter()

for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 4)))

    

for key, value in count.most_common(10):

    print(key, value)
count = Counter()

for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 1)))

    

for key, value in count.most_common(10):

    print(key, value)
count = Counter()

for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 5)))

    

for key, value in count.most_common(10):

    print(key, value)
data.head()
result_product = data.groupby('Product')['Quantity_Ordered'].sum()

result_product
result_product = data.groupby('Product')

quantity_ordered = result_product.sum()['Quantity_Ordered']

products = [product for product, df in result_product]



plt.bar(products, quantity_ordered)

plt.ylabel("Num of Ordered")

plt.xlabel("Product Name")

plt.xticks(products, rotation='vertical')

plt.show()
prices = data.groupby('Product')['Price_Each'].mean()



fig, ax1 = plt.subplots()



ax2 = ax1.twinx()



ax1.bar(products, quantity_ordered, color='g')

ax2.plot(products, prices, 'b-')



ax1.set_xlabel('Product Name')

ax1.set_ylabel('Quantity Ordered', color='g')

ax2.set_ylabel('Price ($)', color='b')

ax1.set_xticklabels(products, rotation='vertical')



plt.show()
# Price High => Quantity Ordered Low

# Price Low => Quantity Ordered High