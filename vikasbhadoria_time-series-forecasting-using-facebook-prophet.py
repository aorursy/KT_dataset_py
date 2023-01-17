!pip install fbprophet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet
# You have to include the full link to the csv file containing your dataset
df_train = pd.read_csv('../input/rossmann-store-sales/train.csv')
df_train.head()
df_train.info()
df_train.describe()
df_store = pd.read_csv('../input/rossmann-store-sales/store.csv')
df_store.head()
# StoreType: categorical variable to indicate type of store (a, b, c, d)
# Assortment: describes an assortment level: a = basic, b = extra, c = extended
# CompetitionDistance (meters): distance to closest competitor store
# CompetitionOpenSince [Month/Year]: provides an estimate of the date when competition was open
# Promo2: Promo2 is a continuing and consecutive promotion for some stores (0 = store is not participating, 1 = store is participating)
# Promo2Since [Year/Week]: date when the store started participating in Promo2
# PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

df_store.info()
df_store.describe()
# Let's see if we have any missing data, luckily we don't!
sns.heatmap(df_train.isnull(),yticklabels=False,cmap='Blues',cbar=False)
df_train.hist(bins=30,color='g',figsize = (20,20))
# Let's see how many stores are open and closed! 
df_train['Open'].value_counts()
# Count the number of stores that are open and closed
print('Total number of stores: {}'.format(len(df_train)))
print('Total number of open stores: 844392')
print('Total number of closed stores: 172817')
# only keep open stores and remove closed stores
df_train = df_train[df_train['Open']==1]
# Let's drop the open column since it has no meaning now
df_train.drop('Open',axis=1,inplace=True)
df_train.shape
df_train.describe()
# Average sales = 6955 Euros,	average number of customers = 762	(went up)
# Let's see if we have any missing data in the store information dataframe!
sns.heatmap(df_store.isnull(),cbar=False,cmap='Blues',yticklabels=False)
# Let's take a look at the missing values in the 'CompetitionDistance'
# Only 3 rows are missing 
df_store['CompetitionDistance'].isnull().sum()
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].mean(),inplace=True)
# Let's take a look at the missing values in the 'CompetitionOpenSinceMonth'
# many rows are missing = 354 (almost one third of the 1115 stores)
df_store['CompetitionOpenSinceMonth'].isnull().sum()
# It seems like if 'promo2' is zero, 'promo2SinceWeek', 'Promo2SinceYear', and 'PromoInterval' information is set to zero
# There are 354 rows where 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' is missing
# Let's set these values to zeros 
df_store.fillna(0,inplace=True)
sns.heatmap(df_store.isnull(),cbar=False,cmap='Blues',yticklabels=False)
# half of stores are involved in promo 2
# half of the stores have their competition at a distance of 0-3000m (3 kms away)
df_store.hist(figsize=(20,20),color='g',bins=30)
# Let's merge both data frames together based on 'store'
merged_df = pd.merge(df_train,df_store,how='inner',on='Store')
merged_df.sample(5)
correlations = merged_df.corr()['Sales'].sort_values(ascending=False)
correlations
# customers and promo are positively correlated with the sales 
# Promo2 does not seem to be effective at all 
correlations = merged_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlations,annot=True,cmap="YlGnBu",linewidths=.5)
# Customers/Prmo2 and sales are strongly correlated 
# Let's separate the year and put it into a separate column 
merged_df['Year'] = pd.DatetimeIndex(merged_df['Date']).year
merged_df['Month'] = pd.DatetimeIndex(merged_df['Date']).month
merged_df['Day'] = pd.DatetimeIndex(merged_df['Date']).day
merged_df.head()
# Let's take a look at the average sales and number of customers per month 
# 'groupby' works great by grouping all the data that share the same month column, then obtain the mean of the sales column  
# It looks like sales and number of customers peak around christmas timeframe
axis = merged_df.groupby('Month')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('Month')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')

# Let's take a look at the sales and customers per day of the month instead
# Minimum number of customers are generally around the 24th of the month 
# Most customers and sales are around 30th and 1st of the month
axis = merged_df.groupby('Day')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('Day')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')

# Let's do the same for the day of the week  (note that 7 = Sunday)
axis = merged_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = merged_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize=(10,5),marker='o',color='g')

fig,ax = plt.subplots(figsize=(20,10))
merged_df.groupby(['Date','StoreType']).mean()['Sales'].unstack().plot(ax=ax)
plt.figure(figsize=(15,10))

plt.subplot(211)
sns.barplot(x="Promo",y='Sales',data=merged_df)
plt.subplot(212)
sns.barplot(x="Promo",y='Customers',data=merged_df)

plt.figure(figsize=(15,10))

plt.subplot(211)
sns.violinplot(x="Promo",y='Sales',data=merged_df)
plt.subplot(212)
sns.violinplot(x="Promo",y='Customers',data=merged_df)
def sales_prediction(Store_ID, sales_df, periods):
  # Function that takes in the data frame, storeID, and number of future period forecast
  # The function then generates date/sales columns in Prophet format
  # The function then makes time series predictions

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')
  
  model    = Prophet()
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods=periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)
sales_prediction(10, merged_df, 60)
def sales_prediction_better(Store_ID, sales_df, holidays, periods):
  # Function that takes in the storeID and returns two date/sales columns in Prophet format
  # Format data to fit prophet 

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')
  
  model    = Prophet(holidays = holidays)
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods = periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)
# Get all the dates pertaining to school holidays 
school_holidays = merged_df[merged_df['SchoolHoliday'] == 1].loc[:, 'Date'].values
school_holidays.shape
# Get all the dates pertaining to state holidays 
state_holidays = merged_df[ (merged_df['StateHoliday'] == 'a') | (merged_df['StateHoliday'] == 'b') | (merged_df['StateHoliday'] == 'c')  ].loc[:, 'Date'].values
state_holidays.shape
state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})
school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                'holiday': 'school_holiday'})
# concatenate both school and state holidays 
school_state_holidays = pd.concat((state_holidays, school_holidays))
# Let's make predictions using holidays for a specific store
sales_prediction_better(14, merged_df, school_state_holidays, 60)
