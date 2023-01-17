# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualisation

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Importing the holidays event data file

holidays_events = pd.read_csv('../input/holidays_events.csv')

holidays_events.describe()
# Importing the items data file

items = pd.read_csv('../input/items.csv')

items.describe(include = 'all')
items.head(10)
items['family'].unique()
item_family = items.groupby(['family', 'class']).item_nbr.size()

item_family.unstack().plot(kind = 'bar', stacked = True, figsize = (13,11), grid = True)
plt.hist(items['perishable'],bins = 4)

plt.title("count of perishable items")

plt.xlabel("perishable")

plt.ylabel("Count")
# Importing the oil data file

oil = pd.read_csv('../input/oil.csv')

oil.describe(include = 'all')
# Replacing the NAN values with the price of oil for the next day

for i in range(0, 1218):

    value = oil['dcoilwtico'][i]

    if np.isnan(oil['dcoilwtico'][i]) == 1:

        value = oil['dcoilwtico'][i+1]

        oil['dcoilwtico'][i] = value
oil.describe(include = 'all')
import datetime

#year = datetime.date.today().year

plt.plot(oil.iloc[:, -1].values, oil.iloc[:, 1].values, color = 'blue')

#plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Oil Price trend')

plt.xlabel('Price')

plt.ylabel('Year')

plt.show()
# Importing the stores data file

stores = pd.read_csv('../input/stores.csv')

stores.describe(include = 'all')
# Stores per state

stores['state'].unique()

state_cluster = stores.groupby(['state', 'cluster']).store_nbr.size()

state_cluster.unstack().plot(kind = 'bar', stacked = True, figsize=(13,11), grid = True)
plt.style.use('seaborn-white')

city_cluster = stores.groupby(['city','type']).store_nbr.size()

city_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis', figsize=(13,11),  grid=False)
plt.style.use('seaborn-white')

city_stores_cluster = stores.groupby(['state', 'city']).store_nbr.size()

city_stores_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis', figsize=(13,11),  grid=False)
# Importing the transactions data file

transactions = pd.read_csv('../input/transactions.csv')

transactions.describe(include = 'all')
transactions.head(10)
X = transactions.iloc[:, 0:2].values

y = transactions.groupby('date').transactions.sum()

print(y)

plt.hist(y)

train = pd.read_csv("../input/train.csv", nrows=30000, parse_dates=['date'])

train.head(10)
# splitting the date into year month and date

next(train)

x = [date  for date in train]

for date_str in x:

    year, month, day = date_str.split('-')

    print(year, month, day)

    