# import libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import missingno

from datetime import datetime
#load the data sets

transaction = pd.read_csv("../input/quantium-data-analytics-virtual-experience-program/Transactions.csv")

purchase = pd.read_csv("../input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv")
#take a first look at the data set from transaction

transaction.head()
#check data types

transaction.dtypes
#check if any missing entry

transaction.isnull().sum()
#check missing entry using visualization: no white lines, so no missing entry

missingno.matrix(transaction, figsize=(15,8))
#look at the first 20 product names (114 total)

transaction["PROD_NAME"].unique()[:20]
#Take a look at all product names with "salsa" in it.

with_salsa_name = transaction[transaction['PROD_NAME'].str.contains('salsa', case=False)]['PROD_NAME'].unique()

print(with_salsa_name)
#List of indices of salsa products

salsa_index = transaction.index[(transaction['PROD_NAME'].str.contains('salsa',case=False)

                          & (transaction['PROD_NAME'].str.contains('Old',case=False)

                             | transaction['PROD_NAME'].str.contains('Woolworths',case=False)))]

#Drop salsa products

chips_tran = transaction.drop(salsa_index, axis=0).reset_index(drop=True)

print("Transactions without Salsa: " + str(len(chips_tran)))
# Add in brand Column

chips_tran['BRAND'] = chips_tran['PROD_NAME'].apply(lambda x : x.strip().split()[0])

brands = {'Dorito':'Doritos','Infzns':'Infuzions',

          'Snbts':'Sunbites','Grain':'Grain Wave',

          'RRD':'Red Rock Deli','Smith':'Smiths',

          'GrnWves':'Grain Wave','WW':'Woolworths',

          'NCC':'Natural','Red':'Red Rock Deli'}

chips_tran['BRAND'] = chips_tran['BRAND'].replace(brands)

chips_tran.head()
#Check if any brand is repeated.

chips_tran.BRAND.value_counts().sort_index()
#total number of brands

len(chips_tran["BRAND"].unique())
#Clean up product names by removing brand names from it.

chips_tran['PROD_NAME'] = chips_tran['PROD_NAME'].apply(lambda x: x.split(' ', 1)[1])

chips_tran.head()
#Define a function to get product packet size

def get_size(item):

    size=[]

    for i in item:

        if i.isdigit():

            size.append(i)

    return int("".join(size))



chips_tran['PKG_SIZE'] = chips_tran['PROD_NAME'].apply(lambda x: get_size(x))



#Look at the table with added column about package size

chips_tran.head()
#Find product unit price and add it as a column to the table

chips_tran['UNIT_PRICE'] = chips_tran['TOT_SALES']/chips_tran['PROD_QTY']

chips_tran.head()
#check the size of the table

chips_tran.shape
#Again look at data types and if there's empty/missing entry.

chips_tran.info()
#data summary from transaction table

chips_tran.describe()
#Use boxplot to see if there is any outlier.

sns.boxplot(chips_tran.PROD_QTY)
#Outliers! Remove them and plot again

tran = chips_tran[chips_tran.PROD_QTY < 10]

print(tran.shape)

sns.boxplot(tran.PROD_QTY)
#look at product quantity summary after removing outliers

tran.PROD_QTY.describe()
#look at total sales summary after removing outliers

tran.TOT_SALES.describe()
#Notice column "DATE" is not in standard form.

tran['DATE'].head()
#define a function to convert the dates to standard form: YYYY-MM-DD

def convert_to_datetime(num):

    dt = datetime.fromordinal(datetime(1900,1,1).toordinal() + num - 2)

    return dt



#Convert dates to form YYYY-MM-DD

tran['DATE'] = tran['DATE'].apply(convert_to_datetime)

tran.head()
#Look at the range of dates

print(tran.DATE.min(), tran.DATE.max())
#Use histogram to look at the distribution of transactions by date: balanced.

sns.set_style('whitegrid')

tran.DATE.hist(figsize=(10,6))
#display the unique number from DATE, STORE_NBR, LYLTY_CARD_NBR, TXN_ID, PROD_NBR columns

#Notice there are repeatitions

print(tran.DATE.nunique())

print(tran.STORE_NBR.nunique())

print(tran.LYLTY_CARD_NBR.nunique())

print(tran.TXN_ID.nunique())

print(tran.PROD_NBR.nunique())
#One day is missing, use line chart to find the missing day.

graph = tran[['DATE', 'TXN_ID']].groupby('DATE').count().sort_values(by='DATE')

ax = graph.plot(figsize=(12,6))

plt.show()
#Find the missing date(s) - Christmas!

from datetime import date, timedelta

dates = sorted(tran.DATE)

date_set = set(dates[0] + timedelta(x) for x in range((dates[-1] - dates[0]).days))

missing = date_set - set(dates)

print(missing)
#Another way to find the missing day

dategroup = tran.groupby('DATE')[['TXN_ID']].count()

pd.date_range(start = '2018-07-01', end = '2019-06-30').difference(dategroup.index)
#A visualization: transaction over time

import plotly.express as px

dategroup = dategroup.reindex(pd.date_range("2018-07-01", "2019-06-30"), fill_value=0)

dategroup['TXN_ID'] = dategroup['TXN_ID'].astype('int')

px.line(dategroup, dategroup.index, dategroup['TXN_ID'])
#Review the clean transaction table.

tran.head()
#look at store numbers (recall there are 272 distinct store numbers)

tran.STORE_NBR.hist()
#Now take a look at some duplicated transaction IDs.

tran[tran.duplicated(['TXN_ID'])].head()
#Look at the products in TXN_ID being 48887, and there are two products.

tran.loc[tran['TXN_ID']==48887, :]
#look at PROD_NBR

tran.PROD_NBR.hist()
#total sales histogram

sns.distplot(tran.TOT_SALES, kde=False)
#look at unit price distribution

sns.distplot(tran.UNIT_PRICE)
#Counts of different brands according to packet size

tran.groupby(['BRAND'])['PKG_SIZE'].value_counts()
#Plot data by brands, 'deep' for categorical variables.

fig,ax = plt.subplots(figsize=(20,18))

plt.subplot(2,1,1)

sns.countplot(tran['BRAND'],palette='deep').set(ylabel='TRANSACTIONS')

plt.title("Transactions of Different Brands")



#Plot data by package size, 'rocket' for quantitative variables.

fig,ax = plt.subplots(figsize=(20,18))

plt.subplot(2,1,2)

sns.countplot(tran['PKG_SIZE'],palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Transactions of Different Package Sizes")
#Take a quick look at the purchase data table

purchase.head()
#look at data types

purchase.info()
#check size of the table

purchase.shape
#check total distinct entries in the 3 columns

purchase.nunique()
#compare LYLTY_CARD_NBR from both original data sets

set(purchase.LYLTY_CARD_NBR.unique()) == set(transaction.LYLTY_CARD_NBR.unique())
#plot lifestage data to check distribution

plt.figure(figsize=(18,8))

sns.countplot(purchase['LIFESTAGE'],palette='rocket_r').set(ylabel='LIFESTAGE')

plt.title("Purchases of Different LIFESTAGES", {'fontsize':15})
#Visualize the premium customers using pie chart and bar graph.

fig,axarr = plt.subplots(1, 2, figsize=(10,5))

purchase['PREMIUM_CUSTOMER'].value_counts().plot.pie(ax=axarr[0])

purchase['PREMIUM_CUSTOMER'].value_counts().plot.bar(ax=axarr[1])
#Combine LIFESTAGE and PREMIUM_CUSTOMER for comparison.

plt.figure(figsize=(20,8))

sns.countplot(purchase['LIFESTAGE'],palette='deep',hue=purchase['PREMIUM_CUSTOMER'])

plt.title("Total Number of Customers by Groups", {'fontsize':15})
#merge data from the two tables (with outliers removed) - left join.

df = tran.merge(purchase, how='left', on='LYLTY_CARD_NBR')

df.shape
#Look at the first 5 rows of the merged table.

df.head()
#Find the total sales, total product quantities, and number of customers for each group.

df_total = df.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'TOT_SALES':'sum',

                        'PROD_QTY':'sum','TXN_ID':'count'}).reset_index()



#Group lifestage and premium customer colomns to form a new column "GROUP".

df_total['GROUP'] = df_total.LIFESTAGE + '_' + df_total.PREMIUM_CUSTOMER

df_total.head()
#Use bar graph to compare total sales.

df_sales = df_total.sort_values('TOT_SALES')

plt.figure(figsize=(20,8))

sns.barplot(x='LIFESTAGE',y='TOT_SALES',hue='PREMIUM_CUSTOMER',data=df_sales)

plt.title("Total Sales by Lifestages and Premium Types", {'fontsize':15})
#Create 3 horizontal bar graphs to display total sales, 

#product quantities, and number of customers by group.

fig, ax = plt.subplots(figsize=(20,30))

plt.subplot(3,1,1)

df_TA = df_total.sort_values('TOT_SALES')

sns.barplot(y='GROUP', x='TOT_SALES', data=df_TA, orient='h', palette='rocket_r')

plt.title("Total Sales by Customer Groups")



plt.subplot(3,1,2)

df_PQ = df_total.sort_values('PROD_QTY')

sns.barplot(y='GROUP', x='PROD_QTY', data=df_PQ, orient='h', palette='rocket_r')

plt.title("Total Quantities Purchased by Customer Groups")



plt.subplot(3,1,3)

df_TI = df_total.sort_values('TXN_ID')

sns.barplot(y='GROUP', x='TXN_ID', data=df_TI, orient='h', palette='rocket_r')

plt.title("Total Number of Customers by Groups")
#Add three more numerical columns.

#Add a new column for sales per customer.

df_total['SALES_PC'] = df_total.TOT_SALES / df_total.TXN_ID



#Add a new column for quantities purchased per customer.

df_total['QTY_PC'] = df_total.PROD_QTY / df_total.TXN_ID



#Add a new column for average dollar amount per bag of chips by group.

df_total['AVG_PQ'] = df_total.TOT_SALES / df_total.PROD_QTY



df_total.head()
#Look at simple numerical summaries of the table.

df_total.describe()
#Create 3 horizontal bar graphs to display total sales, 

#product quantities, and number of customers by group.

fig, ax = plt.subplots(figsize=(20,30))

plt.subplot(3,1,1)

df_SP = df_total.sort_values('SALES_PC')

sns.barplot(y='GROUP', x='SALES_PC', data=df_SP, orient='h', palette='rocket_r')

plt.title("Avarage Sales per Customer by Groups")



plt.subplot(3,1,2)

df_QP = df_total.sort_values('QTY_PC')

sns.barplot(y='GROUP', x='QTY_PC', data=df_QP, orient='h', palette='rocket_r')

plt.title("Average Quantities Purchased per Customer by Groups")



plt.subplot(3,1,3)

df_AP = df_total.sort_values('AVG_PQ')

sns.barplot(y='GROUP', x='AVG_PQ', data=df_AP, orient='h', palette='rocket_r')

plt.title("Average Unit Price of Chips by Groups")