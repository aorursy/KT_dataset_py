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
data_path = '/kaggle/input/open-2-shopee-code-league-order-brushing/order_brush_order.csv'
test_data = pd.read_csv(data_path)
test_data.head()

test_data

#convert field type to datetime 
test_data['event_time'] = pd.to_datetime(test_data['event_time'])
from datetime import datetime 
from datetime import timedelta
thres = 3 #threshold

test_data.dropna(inplace = True)

result = {}

#loop through shops
for shop in test_data.shopid.unique():
    trans = test_data.loc[test_data["shopid"] == shop] #Get all transactions of particular shop 
    res = []
    
    #Ignore shops with too litte transactions 
    if len(trans) >= thres:
        for index, row in trans.iterrows(): #Go through every transactions
            #Get time frame of 1 hour 
            s_date = row['event_time']
            e_date = s_date + timedelta(hours = 1)
            #Transactions within timeframe
            sub_trans = trans.loc[(trans['event_time'] >= s_date) & (trans['event_time'] < e_date)]
            
            #Calculate rate: Count of transactions with timeframe / unique users count
            sub_trans_count = len(sub_trans)
            usr_count = len(sub_trans.userid.unique())
            
            rate = sub_trans_count / usr_count
            
            if rate >= thres:
                usr_trans = {usr: len(sub_trans.loc[sub_trans["userid"] == usr]) for usr in sub_trans.userid.unique()}
                
                #usr_max = max(usr_trans, key = usr_trans.get)
                #fault_usrs = [x for x, v in usr_trans.items() if v == usr_trans[usr_max]]
                
                #may be, use mean instead
                usr_mean = np.mean(list(usr_trans.values()))
                fault_usrs = [x for x, v in usr_trans.items() if v >= usr_trans[usr_mean]]
                
                res.extend([f_usr for f_usr in fault_usrs if f_usr not in res])
                #res.extend(sub_trans.mode().dropna()["userid"])
    result[shop] = sorted(res)
print("fin")
res2 = {}
for key in result:
    users = result[key]
    res2[key] = "0" if len(users) == 0 else "&".join(str(int(x)) for x in users)
output = pd.DataFrame(res2.items(), columns=['shopid', 'userid'])
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
result
