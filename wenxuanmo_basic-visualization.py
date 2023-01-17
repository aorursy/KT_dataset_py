%matplotlib inline
import pandas as pd

from pandas.tools import plotting

import numpy as np

import random



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#%matplotlib nbagg



import time

import sys

from datetime import date, datetime
#Import sales, store

sales = pd.read_csv('../input/train.csv', dtype={'StateHoliday': np.character})

stores = pd.read_csv('../input/store.csv')



print("Sales shape: %s --> columns: %s" % (sales.shape, ", ".join(sales.columns)))

print("Store shape: %s --> columns: %s" % (stores.shape, ", ".join(stores.columns)))
sf = pd.merge(sales, stores, on='Store')

#print "Sales & Store joined shape: %s" % (sf.shape, )



sf['Open'] = sf['Open'].apply(lambda x: 0 if np.isnan(x) else x)



sf['has_sales'] = sf['Sales'].apply(lambda x: x > 0)

sf['has_customers'] = sf['Customers'].apply(lambda x: x != 0)



date_format = '%d/%m/%Y'

if '-' in sf.at[0, 'Date']:

    date_format = '%Y-%m-%d'



    #temporary column that have the date parsed in order to speed up further processing

    sf['_date'] = sf['Date'].apply(lambda x: datetime.strptime(x, date_format))



    sf['month'] = sf['_date'].apply(lambda x: x.month)

    sf['week_of_month'] = sf['_date'].apply(lambda x: x.isocalendar()[1])

    sf['year'] = sf['_date'].apply(lambda x: x.isocalendar()[0])
g_open_cust = sf.groupby(['Open', 'has_customers'])



print("Stores Open/Closed")

print(g_open_cust.agg({'Sales': np.sum, 'Open': np.sum}))
ts = sf.groupby(['StoreType', 'Assortment']).Sales.mean()

ts.plot('bar')

type_assortments = stores.groupby(['StoreType', 'Assortment']).Store.count()
dow = sf[(sf['Sales']!=0)].groupby(['DayOfWeek']).Sales.mean()

dow.plot('bar')
_, ax = plt.subplots(2,2)

sf[(sf['Sales']!=0) & (sf['Store']==907)].boxplot(ax=ax[0][0], column='Sales', by='DayOfWeek')

sf[(sf['Sales']!=0) & (sf['Store']==905)].boxplot(ax=ax[0][1], column='Sales', by='DayOfWeek')

sf[(sf['Sales']!=0) & (sf['Store']==1025)].boxplot(ax=ax[1][0], column='Sales', by='DayOfWeek')

sf[(sf['Sales']!=0) & (sf['Store']==974)].boxplot(ax=ax[1][1], column='Sales', by='DayOfWeek')
stores['CompetitionDistance'].hist(bins=100)
_, ax = plt.subplots(3, 3)

row = 0

col=0

for s,a in type_assortments.keys():

    g = sf[(sf['StoreType'] == s) & (sf['Assortment'] == a)].groupby('month')

    g.Sales.mean().plot('bar', by='month', ax=ax[row/3][col%3], figsize=(9,6))

    row+=1

    col+=1

plt.show()
sns.lmplot(x='week_of_month', y='Sales', data=sf[(sf['Store']==974) & (sf['Sales'] !=0)], 

           col='DayOfWeek', col_wrap=2);
sns.lmplot(x='week_of_month', y='Sales', data=sf[(sf['Store']==907) & (sf['Sales'] !=0)], 

           col='DayOfWeek', col_wrap=3, order=3);
store = 498

data = sf[(sf['Store']==store) & (sf['Sales'] !=0)]

ds = data.groupby(['year', 'month']).agg({'Sales': np.mean}).unstack(['year', 'month']).to_frame()



#reset index so year/month become columns

ds = ds.reset_index([0,1,2]).drop('level_0', axis=1)

ds.columns = ['year', 'month', 'Sales']



_, ax = plt.subplots(2, 2)

ds[ds.year==2013].Sales.plot(ax=ax[0][0])

ds[ds.year==2014].Sales.plot(ax=ax[0][1])

ds[ds.year==2015].Sales.plot(ax=ax[1][0])



sns.lmplot(x='month', y='Sales', data=ds, col='year', col_wrap=3, order=1);
store, month, year = 745, 5, 2015

g_sym = sf.groupby(['Store', 'year', 'month'])

g_store = g_sym.get_group((store, year, month))[['DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo']]

#remove all 0 sales, closed store and reshape

d = g_store[ (g_store.DayOfWeek != 7) & (g_store.Open != 0)].groupby(['DayOfWeek', 'Promo']).Sales.mean().to_frame()

d = d.reset_index()

sns.barplot(data = d, x='DayOfWeek', y='Sales', hue='Promo')
store, month, year = 1000, 5, 2015

g_sym = sf.groupby(['Store', 'year'])

g_store = g_sym.get_group((store, year))[['DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo']]

#remove all 0 sales, closed store and reshape

d = g_store[ (g_store.DayOfWeek != 7) & (g_store.Open != 0)].groupby(['DayOfWeek', 'Promo']).Sales.mean().to_frame()

d = d.reset_index()

sns.barplot(data = d, x='DayOfWeek', y='Sales', hue='Promo')
# Plot sales via date of week on both promo/non-promo days

sns.lmplot(data = d, x='DayOfWeek', y='Sales', col='Promo')
# kernel density for that given store based on DayOfWeek

sns.kdeplot(d.Sales, d.DayOfWeek)