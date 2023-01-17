%matplotlib inline
# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
# get rossmann, store, & test csv files as a DataFrame

rossmann_df  = pd.read_csv("../input/train.csv")

store_df     = pd.read_csv("../input/store.csv")

test_df      = pd.read_csv("../input/test.csv")



# preview the data

rossmann_df.head()
rossmann_df.info()

print("----------------------------")

store_df.info()

print("----------------------------")

test_df.info()
# Open

fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

sns.countplot(x='Open',hue='DayOfWeek', data=rossmann_df,palette="husl", ax=axis1)



# fill NaN values in test_df with Open=1 if DayOfWeek != 7

test_df["Open"][test_df["Open"] != test_df["Open"]] = (test_df["DayOfWeek"] != 7).astype(int)



# Drop Open column

# rossmann_df.drop("Open", axis=1, inplace=True)

# test_df.drop("Open", axis=1, inplace=True)
# Date



# Create Year and Month columns

rossmann_df['Year']  = rossmann_df['Date'].apply(lambda x: int(str(x)[:4]))

rossmann_df['Month'] = rossmann_df['Date'].apply(lambda x: int(str(x)[5:7]))



test_df['Year']  = test_df['Date'].apply(lambda x: int(str(x)[:4]))

test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))



# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)

# this column will be useful in analysis and visualization

rossmann_df['Date'] = rossmann_df['Date'].apply(lambda x: (str(x)[:7]))

test_df['Date']     = test_df['Date'].apply(lambda x: (str(x)[:7]))



# group by date and get average sales, and precent change

average_sales    = rossmann_df.groupby('Date')["Sales"].mean()

pct_change_sales = rossmann_df.groupby('Date')["Sales"].sum().pct_change()



fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))



# plot average sales over time(year-month)

ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")

ax1.set_xticks(range(len(average_sales)))

ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)



# plot precent change for sales over time(year-month)

ax2 = pct_change_sales.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales Percent Change")

# ax2.set_xticks(range(len(pct_change_sales)))

# ax2.set_xticklabels(pct_change_sales.index.tolist(), rotation=90)
# .... contiune with Date



# Plot average sales & customers for every year

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Year', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='Year', y='Customers', data=rossmann_df, ax=axis2)



# Drop Date column

# rossmann_df.drop(['Date'], axis=1,inplace=True)

# test_df.drop(['Date'], axis=1,inplace=True)
# Customers



fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))



# Plot max, min values, & 2nd, 3rd quartile

sns.boxplot([rossmann_df["Customers"]], whis=np.inf, ax=axis1)



# group by date and get average customers, and precent change

average_customers      = rossmann_df.groupby('Date')["Customers"].mean()

# pct_change_customers = rossmann_df.groupby('Date')["Customers"].sum().pct_change()



# Plot average customers over the time

# it should be correlated with the average sales over time

ax = average_customers.plot(legend=True,marker='o', ax=axis2)

ax.set_xticks(range(len(average_customers)))

xlabels = ax.set_xticklabels(average_customers.index.tolist(), rotation=90)
# DayOfWeek

# In both cases where the store is closed and opened



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='DayOfWeek', y='Sales', data=rossmann_df, order=[1,2,3,4,5,6,7], ax=axis1)

sns.barplot(x='DayOfWeek', y='Customers', data=rossmann_df, order=[1,2,3,4,5,6,7], ax=axis2)
# Promo



# Plot average sales & customers with/without promo

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Promo', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='Promo', y='Customers', data=rossmann_df, ax=axis2)
# StateHoliday



# StateHoliday column has values 0 & "0", So, we need to merge values with 0 to "0"

rossmann_df["StateHoliday"].loc[rossmann_df["StateHoliday"] == 0] = "0"

# test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"



# Plot

sns.countplot(x='StateHoliday', data=rossmann_df)



# Before

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StateHoliday', y='Sales', data=rossmann_df, ax=axis1)



mask = (rossmann_df["StateHoliday"] != "0") & (rossmann_df["Sales"] > 0)

sns.barplot(x='StateHoliday', y='Sales', data=rossmann_df[mask], ax=axis2)
# .... continue with StateHoliday



# After

rossmann_df["StateHoliday"] = rossmann_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

test_df["StateHoliday"]     = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StateHoliday', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='StateHoliday', y='Customers', data=rossmann_df, ax=axis2)
# SchoolHoliday



# Plot

sns.countplot(x='SchoolHoliday', data=rossmann_df)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='SchoolHoliday', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='SchoolHoliday', y='Customers', data=rossmann_df, ax=axis2)
# Sales



fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))



# Plot max, min values, & 2nd, 3rd quartile

sns.boxplot([rossmann_df["Customers"]], whis=np.inf, ax=axis1)



# Plot sales values 

# Notice that values with 0 is mostly because the store was closed

rossmann_df["Sales"].plot(kind='hist',bins=70,xlim=(0,15000),ax=axis2)
# Using store_df



# Merge store_df with average store sales & customers

average_sales_customers = rossmann_df.groupby('Store')[["Sales", "Customers"]].mean()

sales_customers_df = DataFrame({'Store':average_sales_customers.index,

                      'Sales':average_sales_customers["Sales"], 'Customers': average_sales_customers["Customers"]}, 

                      columns=['Store', 'Sales', 'Customers'])

store_df = pd.merge(sales_customers_df, store_df, on='Store')



store_df.head()
# StoreType 



# Plot StoreType, & StoreType Vs average sales and customers



sns.countplot(x='StoreType', data=store_df, order=['a','b','c', 'd'])



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StoreType', y='Sales', data=store_df, order=['a','b','c', 'd'],ax=axis1)

sns.barplot(x='StoreType', y='Customers', data=store_df, order=['a','b','c', 'd'], ax=axis2)
# Assortment 



# Plot Assortment, & Assortment Vs average sales and customers



sns.countplot(x='Assortment', data=store_df, order=['a','b','c'])



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Assortment', y='Sales', data=store_df, order=['a','b','c'], ax=axis1)

sns.barplot(x='Assortment', y='Customers', data=store_df, order=['a','b','c'], ax=axis2)
# Promo2



# Plot Promo2, & Promo2 Vs average sales and customers



sns.countplot(x='Promo2', data=store_df)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Promo2', y='Sales', data=store_df, ax=axis1)

sns.barplot(x='Promo2', y='Customers', data=store_df, ax=axis2)
# CompetitionDistance



# fill NaN values

store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())



# Plot CompetitionDistance Vs Sales

store_df.plot(kind='scatter',x='CompetitionDistance',y='Sales',figsize=(15,4))

store_df.plot(kind='kde',x='CompetitionDistance',y='Sales',figsize=(15,4))
# What happened to the average sales of a store over time when competition started?

# Example: the average sales for store_id = 6 has dramatically decreased since the competition started



store_id = 6

store_data = rossmann_df[rossmann_df["Store"] == store_id]



average_store_sales = store_data.groupby('Date')["Sales"].mean()



# Get year, and month when Competition started

y = store_df["CompetitionOpenSinceYear"].loc[store_df["Store"]  == store_id].values[0]

m = store_df["CompetitionOpenSinceMonth"].loc[store_df["Store"] == store_id].values[0]



# Plot 

ax = average_store_sales.plot(legend=True,figsize=(15,4),marker='o')

ax.set_xticks(range(len(average_store_sales)))

ax.set_xticklabels(average_store_sales.index.tolist(), rotation=90)



# Since all data of store sales given in rossmann_df starts with year=2013 till 2015,

# So, we need to check if year>=2013 and y & m aren't NaN values.

if y >= 2013 and y == y and m == m:

    plt.axvline(x=((y-2013) * 12) + (m - 1), linewidth=3, color='grey')