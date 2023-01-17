# Importing libraries
import numpy as np
import pandas as pd
import os
from datetime import datetime
from matplotlib import pyplot as plt
# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()
# print(os.listdir("../input"))
data = pd.read_csv('../input/BreadBasket_DMS.csv')
print(data.info())
print(data.head())
# Join Date and Time column
data['datetime_combined'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)

# Convert the new column into datetime column
data['datetime_combined'] = data['datetime_combined'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
print(type(data['datetime_combined'][0]))

# What's the date range of this dataset?
print(data.datetime_combined.min(), data.datetime_combined.max())
print("Interval: ", data.datetime_combined.max() - data.datetime_combined.min())
item_list = data.Item.unique()
print("Number of unique items: ", len(item_list))
#print(item_list)
item_count = data['Item'].value_counts().reset_index()
item_count.columns = ['Item', 'Count']
top5_list = item_count['Item'][:5].tolist()
print("Top 5 items: ", top5_list)
plt.figure(figsize=(10,5))
plt.bar(item_count[:5].Item, item_count[:5].Count, color = 'blue')
plt.show()
# Overall trend, group by date.
# We have to make sure to not count repeated transaction number
new_df = data.groupby(['Transaction', 'Date']).count()
new_df = new_df.groupby(['Date']).size().reset_index()
new_df = pd.DataFrame(new_df)
new_df.columns = ['Date', 'Count']
new_df.Date = new_df.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) 
new_df = new_df.set_index('Date')

# Visualization
plt.figure(figsize=(15,5))
plt.xlabel('Date')
plt.ylabel('Number of transactions')
plt.title('Overall Sales trend')
plt.plot(new_df.Count)
plt.show()
# Overall trend, group by date.
# We have to make sure to not count repeated transaction number
new_df = data.loc[data['Item']=='Bread']
new_df = new_df.groupby(['Transaction', 'Date']).count()
new_df = new_df.groupby(['Date']).size().reset_index()
new_df = pd.DataFrame(new_df)
new_df.columns = ['Date', 'Count']
new_df.Date = new_df.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) 
new_df = new_df.set_index('Date')

# Visualization
plt.figure(figsize=(15,5))
plt.xlabel('Date')
plt.ylabel('Number of transactions')
plt.title('Overall Sales trend')
plt.plot(new_df.Count)
plt.show()
# Let's obtain the day of week from the datetime value we have
data['day_of_week'] = data['datetime_combined'].apply(lambda x: x.strftime('%A'))
transaction_by_day = data.groupby(['Transaction', 'day_of_week']).count().reset_index()
transaction_by_day = transaction_by_day[['Transaction', 'day_of_week']]

# Count the number of transactions made on each day
transaction_by_day = transaction_by_day.groupby(['day_of_week']).count().reset_index()

# Sort by day of week (Sunday, Monday, Tuesday ...) http://blog.quizzicol.com/2016/10/03/sorting-dates-in-python-by-day-of-week/
sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sorterIndex = dict(zip(sorter,range(len(sorter))))
transaction_by_day['Day_id'] = transaction_by_day.day_of_week
transaction_by_day['Day_id'] = transaction_by_day['Day_id'].map(sorterIndex)
transaction_by_day.sort_values('Day_id', inplace=True)
plt.figure(figsize=(10,5))
plt.xlabel('Day of Week')
plt.ylabel('Number of transactions')
plt.title('Sales made on different days')
plt.bar(transaction_by_day.day_of_week, transaction_by_day.Transaction, color = 'blue')
plt.show()
# Getting the 'hour' from datetime
data['Hour'] = data.datetime_combined.dt.hour
# data.head()

# Let's count the total transactions for each hour
transaction_by_hour = data.groupby(['Transaction', 'Hour']).count().reset_index()
transaction_by_hour= transaction_by_hour[['Transaction', 'Hour']]
# transaction_by_hour.head()

transaction_by_hour = transaction_by_hour.groupby(['Hour']).count().reset_index()
# Overall sales trend based on hours
plt.figure(figsize=(10,5))
plt.xlabel('Hour')
plt.ylabel('Number of transactions')
plt.title('Sales trend by Hour')
plt.xticks(np.arange(0, 23, step=1))
plt.plot(transaction_by_hour.Hour, transaction_by_hour.Transaction)
plt.show()

top5_items = data.loc[data.Item.isin(top5_list)]
transaction_by_item = pd.crosstab(top5_items.Hour, top5_items.Item)

# Sales trend of top 5 sold items
plt.figure(figsize=(10,5))
plt.xlabel('Hour')
plt.ylabel('Number of Transaction')
plt.title("Top 5 Sold Items' Sales trend by Hour")
for item in top5_list:
    plt.plot(transaction_by_item.index, transaction_by_item[item])
    plt.legend(top5_list)
plt.show()
# Another method, visualizing the trends separately 
transaction_by_item.plot(subplots=True)
plt.show()
# Let's obtain the day of week from the datetime value we have
transaction_saturday = data.loc[data['day_of_week']=='Saturday']
transaction_saturday = transaction_saturday.groupby(['Transaction', 'day_of_week', 'Hour', 'Item']).count().reset_index()
transaction_saturday = transaction_saturday[['Transaction', 'day_of_week', 'Hour', 'Item']]
transaction_saturday_top5 = transaction_saturday.loc[transaction_saturday.Item.isin(top5_list)]
transaction_saturday_top5 = pd.crosstab(transaction_saturday_top5.Hour, transaction_saturday_top5.Item)
transaction_saturday_top5.plot(subplots=True)
plt.show()