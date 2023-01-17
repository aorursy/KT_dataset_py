# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
order_data = pd.read_csv('../input/students-order-brushing-1/order_brush_order.csv')

order_data.head()
order_data['event_time'] = pd.to_datetime(order_data.event_time)

order_data.dtypes
def get_suspicious_buyer(df):

    df.sort_values(by='event_time', inplace=True)

#     print(df, end='\n\n')

    

    n = len(df.index)

    is_suspicious = [False for _ in range(n)]

        

    for i in range(n):

        maxJ = -1

        userid_set = set()        

        for j in range(i, n):

            delta_second = (df['event_time'].iloc[j] - df['event_time'].iloc[i]).total_seconds()

            if delta_second > 3600:

                break

            userid_set.add(df['userid'].iloc[j])

            if j-i+1 >= len(userid_set) * 3:

                maxJ = j            

        for j in range(i, maxJ+1):

            is_suspicious[j] = True

            

    brush_df = df.loc[is_suspicious]

#     print(brush_df, end='\n\n')

    

    user_count = brush_df.groupby('userid').orderid.count()

#     print(user_count, end='\n\n')

    

    most_suspicious_users = list(user_count[user_count == user_count.max()].index)

    most_suspicious_users.sort()

    

    res = '&'.join([str(x) for x in most_suspicious_users])

    if res == '':

        res = '0'

    return res
shop_groups = order_data.groupby('shopid')



suspicious_users = []

for shop_id, df in shop_groups:    

    suspicious_users.append(get_suspicious_buyer(df))
shop_ids = []

for shop_id, df in shop_groups:

    shop_ids.append(shop_id)



output = pd.DataFrame({'shopid': shop_ids,

                       'userid': suspicious_users})

output.to_csv('submission.csv', index=False)