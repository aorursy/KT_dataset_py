import pandas as pd

import numpy as np

import datetime

import os
df = pd.read_csv('/kaggle/input/delivery_orders_march.csv')



# changing from epoch time to date strings

df['1st_deliver_attempt'] = df['1st_deliver_attempt'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

df['pick'] = df['pick'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

df['2nd_deliver_attempt'] = df['2nd_deliver_attempt'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else x)





# Converting to datetime objects for all dates 

df['1st_deliver_attempt'] = pd.to_datetime(df['1st_deliver_attempt'])

df['pick'] = pd.to_datetime(df['pick'])

df['2nd_deliver_attempt'] = pd.to_datetime(df['2nd_deliver_attempt'], errors ='coerce')





# Creating Rows to track time difference between deliveries 

df['1st_attempt'] = df['1st_deliver_attempt'] - df['pick']

df['1st_attempt'] = df['1st_attempt'].dt.days.astype(float)

df['2nd_attempt'] = df['2nd_deliver_attempt'] - df['1st_deliver_attempt']

df['2nd_attempt'] = df['2nd_attempt'].dt.days





# Preprocessing addresses 

df['buyeraddress'] = df['buyeraddress'].apply(lambda x : x.split(' ')[-1:]).apply(lambda x : ' '.join(x).lower())

df['selleraddress'] = df['selleraddress'].apply(lambda x : x.split(' ')[-1:]).apply(lambda x : ' '.join(x).lower())





# Replacing 0's for NaN values

df['2nd_attempt'].fillna(0, inplace=True)



df.head()
df.info()
df['is_late'] = [0 if attempt < 7 and attempt2 < 3 else 1 for (attempt, attempt2) in zip(df['1st_attempt'], df['2nd_attempt'])]
df2 = df[['orderid', '1st_attempt', '2nd_attempt', 'buyeraddress', 'selleraddress']]
df2_1 = df2[(df2['buyeraddress'] == 'manila') & (df2['selleraddress'] == 'manila')]

df2_1['is_late'] = [0 if attempt<3 and attempt2<3 else 1 for (attempt,attempt2) in zip(df2_1['1st_attempt'], df2_1['2nd_attempt'])]

df2_1.head() # need to be combined
df2_2 = df2[(df2['buyeraddress'] == 'luzon') & (df2['selleraddress'] == 'manila')]

df2_2['is_late'] = [0 if attempt<5 and attempt2<3 else 1 for (attempt,attempt2) in zip(df2_2['1st_attempt'], df2_2['2nd_attempt'])]

df2_2.head() # needs to be combined
df2_3 = df2[(df2['buyeraddress'] == 'manila') & (df2['selleraddress'] == 'luzon')]

df2_3['is_late'] = [0 if attempt<5 and attempt2<3 else 1 for (attempt,attempt2) in zip(df2_3['1st_attempt'], df2_3['2nd_attempt'])]

df2_3 # nothing for this condition
df2_4 = df2[(df2['buyeraddress'] == 'luzon') & (df2['selleraddress'] == 'luzon')]

df2_4['is_late'] = [0 if attempt<5 and attempt2<3 else 1 for (attempt,attempt2) in zip(df2_4['1st_attempt'], df2_4['2nd_attempt'])]

df2_4 # needs to be combined
df_partial = pd.concat([df2_1, df2_2, df2_4])

df_partial
df2['is_late'] = [0 if attempt < 7 and attempt2 < 3 else 1 for (attempt, attempt2) in zip(df2['1st_attempt'], df2['2nd_attempt'])]

df2 # Rest of the conditions
df_full = pd.merge(df_partial, df2, on='orderid', how='outer')
df_full
df_full.is_late_x = df_full.is_late_x.fillna(df_full.is_late_y) # combining values

df_full = df_full[['orderid', 'is_late_x']] # taking the relevant information

df_full = df_full[['orderid', 'is_late_x']].rename(columns={'is_late_x' : 'is_late'})

df_full.is_late = df_full.is_late.astype('int')
df_full
df_full.info()
df_submission = df_full.copy()
df_submission.to_csv('submission.csv', index=False)
df_submission.info()