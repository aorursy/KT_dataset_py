import numpy as np

import pandas as pd

from pandas import Series, DataFrame
# read data from file

df_ad_data = pd.read_csv('../input/homework/pandas/ad/data.txt', sep = ' ', header = 0,index_col = 'instance_id')

df_ad_data.head()
df_ad_data.shape
# group by user_id

df_user_group = df_ad_data.groupby('user_id')
df1 = df_user_group.size()

df1
df2 = df_user_group['is_trade'].sum()

df2
df3 = pd.concat([df1, df2], axis = 1)

df3
df3.columns = ['total_ad_clicks', 'trade_nums']

df3
# User conversion rate dataframe

df_user_conv_rate = df3.assign(user_conv_rate = lambda x: x['trade_nums'] / x['total_ad_clicks'])

df_user_conv_rate
df_user_conv_rate.to_csv('ad-user_conversion_rate.csv')
df_user_conv_rate.describe()
# group by item_id

df_ad_group = df_ad_data.groupby('item_id')

df_ad_group.size()
df4 = df_ad_group.size()

df4
df5 = df_ad_group['is_trade'].sum()

df5
df6 = pd.concat([df4, df5], axis = 1)

df6
df6.columns = ['total_user_clicks', 'trade_nums']

df6
# Ad conversion rate dataframe

df_ad_conv_rate = df6.assign(ad_conv_rate = lambda x: x['trade_nums'] / x['total_user_clicks'])

df_ad_conv_rate
df_ad_conv_rate.describe()
df_ad_conv_rate.to_csv('ad-item_conversion_rate.csv')
# check missing values within a dataframe

df_ad_data.isnull().any()
# check missing values in a Series (one column in a dataframe)

df_ad_data['context_timestamp'].hasnans
# check misssing values in a Series (one row in a dataframe)

df_ad_data.iloc[5].hasnans
df_ad_data_sorted = df_ad_data.sort_values(by = ['item_id', 'user_id', 'context_timestamp'])

df_ad_data_sorted[['item_id', 'user_id', 'context_timestamp']].head(30)
df_ad_data_sorted['time_interval_last_watch'] = 0
# save the previous record, item_id, user_id and context_timestamp.

pre_item_id = df_ad_data_sorted.item_id[0:1].values

pre_user_id = df_ad_data_sorted.user_id[0:1].values

pre_timestamp = df_ad_data_sorted.context_timestamp[0:1].values



print('pre_item_id is ' + str(pre_item_id) + ', pre_user_id is ' + str(pre_user_id) + ', pre_timestamp is ' + str(pre_timestamp) + '.')



# calculate  and save time_interval_last_watch

for index, row in df_ad_data_sorted.iterrows():

    if row['item_id']==pre_item_id and row['user_id']==pre_user_id:

        df_ad_data_sorted.at[index, 'time_interval_last_watch'] = row['context_timestamp'] - pre_timestamp

    else:

        df_ad_data_sorted.at[index, 'time_interval_last_watch'] = 0



    # save item_id, user_id and context_timestamp for next row.

    pre_item_id = row['item_id']

    pre_user_id = row['user_id']

    pre_timestamp = row['context_timestamp']



print('finished')
df_ad_data_sorted.describe()
df_ad_data_sorted.to_csv('ad-last_watch_time_interval.csv')