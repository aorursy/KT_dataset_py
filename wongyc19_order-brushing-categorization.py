%%time 

import pandas as pd
import os
from getpass import getuser
from datetime import datetime

file_path = ""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)

# convert event_time column from string to datetime object
data['event_time'] = pd.to_datetime(data['event_time'])

# sort by seller, buyer, and date - This is important for the following operation
data.sort_values(by=['shopid', 'userid', 'event_time'], ascending=True, inplace=True)

# Convert datetime to UNIX format (number of seconds since 1970/01/01)
data['seconds'] = (data['event_time'] - datetime(1970, 1, 1)).dt.total_seconds()
data.reset_index(drop=True, inplace=True)

keys = data[['shopid', 'userid']].copy()

# Features #1: interval_hour = True if 3 transactions is made consecutively in 1 hour range
interval1 = data['seconds'].diff(1).fillna(0)
interval2 = data['seconds'].diff(2).fillna(0)

separator1 = (keys != keys.shift(1)).sum(axis=1) > 0
separator2 = (keys != keys.shift(2)).sum(axis=1) > 0

interval1[separator1] = 0 
interval2[separator2] = 0 

data['interval_hour'] = (interval1 <= 3600) & (interval2 <= 3600)

# Features #2: batch_order = True if both parties are the same for 3 rows consecutively
shift1 = (keys == keys.shift(1)).sum(axis=1) == 2
shift2 = (keys == keys.shift(2)).sum(axis=1) == 2
data['batch_order'] = shift1 & shift2

data['brush_order'] = (data['interval_hour']) & (data['batch_order'])
brushing = data.loc[data['brush_order'], ['shopid', 'userid']]
brushing.drop_duplicates(inplace=True)
brushing['userid'] = brushing['userid'].astype(str)
brushing_result = brushing.groupby('shopid').agg(lambda x: "&".join(x))

result = pd.DataFrame(data= data['shopid'].unique(), columns=['shopid'])
result = result.merge(brushing_result, left_on='shopid', right_index=True, how='left').fillna("0")

# Get the top transactions from the potential order brushing user
for idx, shop, user_ids in result[result['userid'].str.count('&') > 1].itertuples(True, None):
    user_ids = user_ids.split("&")
    score_per_user = data[(data.userid.isin(user_ids)) & (data.shopid==shop)].groupby('userid')['batch_order'].sum()
    userid = '&'.join(map(str,score_per_user[score_per_user == score_per_user.max()].index))
    result.at[idx, 'userid'] = userid
    
result = result.applymap(str)
result.to_csv('mysubmission.csv', index=False)
print('End of program')