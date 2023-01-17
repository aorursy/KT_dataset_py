import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# required for visualizing plots directly in Jupyter Notebook
%matplotlib inline
# reading the training dataset
train = pd.read_csv('../input/sales_train.csv')

# explore the train dataset structure
train.head()
train.tail()
# filtering the train dataset only on the SALES data
filter_1 = train['item_cnt_day'] > 0

# applying the filter and creating new dataframe
filtered_1 = train[filter_1]

# setting a plot size
plt.rcParams['figure.figsize'] = (40.0, 20.0)

# 2D plot for total sales/month per shops
sns.heatmap(filtered_1.pivot_table(values='item_cnt_day', 
                                   index='shop_id', 
                                   columns='date_block_num', 
                                   aggfunc='sum', 
                                   dropna=True
                                  ), 
            linewidths=1, 
            annot=True, 
            annot_kws={"size": 10}
           )
# list of shops which were operational during the last month 'date_block_num' == 33
filter_2 = filtered_1[(filtered_1['date_block_num'] == 33)]['shop_id'].unique()

# subset of the sales dataset based on filter_1
filtered_2 = filtered_1[filtered_1['shop_id'].isin(filter_2)]
# setting a plot size
plt.rcParams['figure.figsize'] = (40.0, 20.0)

# 2D plot for total sales/month per shops
sns.heatmap(filtered_2.pivot_table(values='item_cnt_day', 
                                   index='shop_id', 
                                   columns='date_block_num', 
                                   aggfunc='sum', 
                                   dropna=True
                                  ), 
            linewidths=1, 
            annot=True, 
            annot_kws={"size": 10}
           )
# defining a list with the month names for a period of 99 year (99 is more or less an arbitrary number)
months = ['January','February','March','April','May','June','July','August','September','October','November','December']*99

# setting a plot size
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# 2D plot for total sales/month per shops
sns.heatmap(filtered_2.pivot_table(values='item_cnt_day', 
                                   index='shop_id', 
                                   columns='date_block_num', 
                                   aggfunc='sum', 
                                   dropna=True
                                  ), 
            linewidths=1, 
            xticklabels=months[0:34]
           )
# list of shops which were operational in 'December'
filter_3 = filtered_2[(filtered_2['date_block_num'] == 23)]['shop_id'].unique()

# subset of the sales dataset based on filter_1
filtered_3 = filtered_2[filtered_2['shop_id'].isin(filter_3)]

# defining a list with the month names for a period of 99 year (99 is more or less an arbitrary number)
months = ['January','February','March','April','May','June','July','August','September','October','November','December']*99

# setting a plot size
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# 2D plot for total sales/month per shops
sns.heatmap(filtered_3.pivot_table(values='item_cnt_day', 
                                   index='shop_id', 
                                   columns='date_block_num', 
                                   aggfunc='sum', 
                                   dropna=True
                                  ), 
            linewidths=1, 
            xticklabels=months[0:34]
           )

# rotating the shop_id labels on the plot
plt.yticks(rotation='horizontal')
# assigning the pivot table into a variable
filtered_3_pivoted = filtered_3.pivot_table(values='item_cnt_day', 
                                   index='shop_id', 
                                   columns='date_block_num', 
                                   aggfunc='sum', 
                                   dropna=True
                                  )

# converting the pivot table into a dataframe
filtered_3_records = pd.DataFrame(filtered_3_pivoted.to_records())

# transposing the dataframe
filtered_3_records = filtered_3_records.transpose()

# assiginig the correct column names
filtered_3_records.columns = filtered_3_records.loc['shop_id'].astype('str')

# ---------------- converting 2.0 into 2 and etc.
new_names = []

for name in filtered_3_records.columns:
    new_names.append(name.split('.')[0])

filtered_3_records.columns = new_names

# removing the 'shop_id' row
filtered_3_records.drop('shop_id', axis=0, inplace=True)

# replacing the NaN values with 0.0
filtered_3_records.fillna(value=0.0, inplace=True)

# converting the index to dtype='int64'
filtered_3_records.index = filtered_3_records.index.astype('int64')

# exploring the dataframe
filtered_3_records.head()
# showing time-history of total sales for shop 31
filtered_3_records['31'].plot()
plt.rcParams['figure.figsize'] = (13, 13)

filtered_3_records['date'] = pd.date_range(start='2013-01-01', end='2015-11-01', freq='M')

filtered_3_records = filtered_3_records.set_index('date')

decomposed = sm.tsa.seasonal_decompose(filtered_3_records['31'], freq=12, model='multiplicative').plot()
residuals = sm.tsa.seasonal_decompose(filtered_3_records['31'], freq=12, model='multiplicative').resid

plt.rcParams['figure.figsize'] = (5, 5)

plt.hist(residuals.dropna(), bins=3)
from fbprophet import Prophet 
df = pd.DataFrame()

df['ds'] = filtered_3_records['31'].index
df['y'] = filtered_3_records['31'].values

m = Prophet().fit(df)
future = m.make_future_dataframe(periods=1, freq='M')
fcst = m.predict(future)
m.plot(fcst)
plt.axvline(x='2015-10-31', color='red')
fcst.head()
fcst[['yhat_lower','yhat','yhat_upper']].loc[32:34]
def prediction(series):
    
    # create an empty dataframe
    df = pd.DataFrame()

    # assign columns
    df['ds'] = series.index
    df['y'] = series.values
    
    # make the actual prediction for the next month
    m = Prophet(weekly_seasonality=False, daily_seasonality=False).fit(df)
    future = m.make_future_dataframe(periods=1, freq='M')
    fcst = m.predict(future)
    
    # return the predicted value for the last month
    return fcst['yhat'].loc[34]
# initializing an empty dictionary to store the predicted values
predicted_sales = {}

for column in filtered_3_records.columns:
    
    # extracting a single column of the dataframe into a series
    # passing the series to our wrapper function
    predicted_value = prediction(filtered_3_records[column])
    
    # storing the predicted value into a dictionary
    predicted_sales[column] = predicted_value
df = pd.DataFrame.from_dict(predicted_sales, orient='index')
df = df.transpose()

# adding a new row to our dataframe
filtered_3_records = filtered_3_records.append(df, ignore_index=True)

# checking the last 5 rows of the dataframe
filtered_3_records.tail()
# subsetting the dataframe to November 2013
November_2013 = filtered_3[filtered_3['date_block_num'] == 10]

# subsetting the dataframe to November 2014
November_2014 = filtered_3[filtered_3['date_block_num'] == 22]

# creating a pivot table 
November_2013_pivoted = November_2013.pivot_table(index='shop_id', columns='item_id', values='item_cnt_day', aggfunc='sum', fill_value=0)

# converting the pivot table into a dataframe
November_2013_records = pd.DataFrame(November_2013_pivoted.to_records())

# setting an index for the dataframe
November_2013_records.set_index('shop_id', inplace=True)

# normalizing the dataframe
# here a convert the number of sales per item to a fraction of the total sales for the shops
November_2013_records = November_2013_records.div(November_2013_records.sum(axis=1), axis=0)

November_2013_records.head()
# creating a pivot table 
November_2014_pivoted = November_2014.pivot_table(index='shop_id', columns='item_id', values='item_cnt_day', aggfunc='sum', fill_value=0)

# converting the pivot table into a dataframe
November_2014_records = pd.DataFrame(November_2014_pivoted.to_records())

# setting an index for the dataframe
November_2014_records.set_index('shop_id', inplace=True)

# normalizing the dataframe
# here a convert the number of sales per item to a fraction of the total sales for the shops
November_2014_records = November_2014_records.div(November_2014_records.sum(axis=1), axis=0)

November_2014_records.head()
# merging the two dataframes and in the same time taking the mean for the columns with common names
# https://stackoverflow.com/questions/50312018/merge-two-dataframes-in-pandas-by-taking-the-mean-between-the-columns
November_concat = pd.concat([November_2013_records, November_2014_records], axis=1).groupby(axis=1, level=0).mean()

# softmax row-wise
November_concat = November_concat.div(November_concat.sum(axis=1), axis=0)

# converting to number of sales
series = filtered_3_records.loc[34]
series.index = series.index.astype('int64')
series.index.name = 'shop_id'

November_concat = November_concat.mul(series, axis=0)

# replace NaN values with 0.0
November_concat.fillna(0.0, inplace=True)

# rounding the sales to whole numbers
November_concat = November_concat.round(decimals=0)

November_concat.head()
test = pd.read_csv('../input/test.csv', index_col='ID')
test.head()
def lookup(df, row):
    '''
    function which returns the total number of sales of given item_id for given shop_id
    '''
    try:
        value = df.loc[row['shop_id']][row['item_id']]
    except:
        # KeyError -> store_id missing
        # IndexError -> item_id missing
        # for both cases makes sense to predict 0 as either the store is closed or the given item was not sold even once
        # in the given store in the past Novembers
        value = 0
    
    return value

# apply the lookup function to each row of the test dataframe and store the result in 'prediction' column
test['item_cnt_month'] = test.apply(lambda row: lookup(November_concat,row), axis=1)
test['item_cnt_month'].clip(lower=0, upper=20, inplace=True)
submission = test.drop(['shop_id','item_id'], axis=1)

submission.to_csv('submission-01.csv')