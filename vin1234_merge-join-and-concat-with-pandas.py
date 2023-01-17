import pandas as pd

import numpy as np
dummy_data1 = {

        'id': ['1', '2', '3', '4', '5'],

        'Feature1': ['A', 'C', 'E', 'G', 'I'],

        'Feature2': ['B', 'D', 'F', 'H', 'J']}



df1 = pd.DataFrame(dummy_data1, columns = ['id', 'Feature1', 'Feature2'])



df1
dummy_data2 = {

        'id': ['1', '2', '6', '7', '8'],

        'Feature1': ['K', 'M', 'O', 'Q', 'S'],

        'Feature2': ['L', 'N', 'P', 'R', 'T']}

df2 = pd.DataFrame(dummy_data2, columns = ['id', 'Feature1', 'Feature2'])



df2
dummy_data3 = {

        'id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],

        'Feature3': [12, 13, 14, 15, 16, 17, 15, 12, 13, 23]}

df3 = pd.DataFrame(dummy_data3, columns = ['id', 'Feature3'])



df3
df_row = pd.concat([df1, df2],ignore_index=True)



df_row
df_merge_col=pd.merge(df_row,df3,on='id')
df_merge_col
df_merge_difkey = pd.merge(df_row, df3, left_on='id', right_on='id')



df_merge_difkey
user_usage=pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/user_usage.csv')

user_usage.head()
# user_usage shape

user_usage.shape
# user device data

user_device=pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/user_device.csv')

user_device.head()
# user_device shape

user_device.shape
device=pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/android_devices.csv')

device.head()
# device shape

device.shape
data1=pd.merge(user_usage,user_device[['use_id', 'platform', 'device']],on='use_id',how='outer',indicator=True)

data1.head()
data1.shape
data1.head()
result = pd.merge(user_usage,

                 user_device[['use_id', 'platform', 'device']],

                 on='use_id')



result.head()
print(result.shape)
result = pd.merge(user_usage,

                 user_device[['use_id', 'platform', 'device']],

                 on='use_id', 

                 how='left')
result.tail()
result = pd.merge(user_usage,

                 user_device[['use_id', 'platform', 'device']],

                 on='use_id', 

                 how='right')
result.head()
result.tail()
result = pd.merge(user_usage,

                 user_device[['use_id', 'platform', 'device']],

                 on='use_id', 

                 how='outer', 

                 indicator=True)
print(user_usage.shape)

print(user_device.shape)
# First, add the platform and device to the user usage - use a left join this time.



result = pd.merge(user_usage,

                 user_device[['use_id', 'platform', 'device']],

                 on='use_id',

                 how='left')

result.head()
result.shape
# Now, based on the "device" column in result, match the "Model" column in devices.

device.rename(columns={"Retail Branding": "manufacturer"}, inplace=True)

result = pd.merge(result, 

                  device[['manufacturer', 'Model']],

                  left_on='device',

                  right_on='Model',

                  how='left')



result.head()