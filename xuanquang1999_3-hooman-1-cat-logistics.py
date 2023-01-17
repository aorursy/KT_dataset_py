# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")
data.head()
addresses = ["metro manila", "luzon", "visayas", "mindanao"]

data["buyeraddress"] = data["buyeraddress"].str.lower()
data["selleraddress"] = data["selleraddress"].str.lower()

for i in range(len(addresses)):
    address = addresses[i]
    data["buyeraddress"][data["buyeraddress"].str.endswith(address, na=False)] = i
    data["selleraddress"][data["selleraddress"].str.endswith(address, na=False)] = i
    
data.head(n=20)
sla_table = [
    [3, 5, 7, 7],
    [5, 5, 7, 7],
    [7, 7, 7, 7],
    [7, 7, 7, 7]
]

sla = []
for i in range(len(data)):
    buyer_add_id = data["buyeraddress"][i]
    seller_add_id = data["selleraddress"][i]
    sla.append(sla_table[seller_add_id][buyer_add_id])
    
data["sla"] = sla
data.head(n=20)
for column in ["pick", "1st_deliver_attempt", "2nd_deliver_attempt"]:
    data[column] = pd.to_datetime(data[column], unit='s')
    data[column] = data[column].values.astype('datetime64[D]')

data.head(n=20)
weekmask = 'Mon Tue Wed Thu Fri Sat'
holidays = ['2020-03-08', '2020-03-25', '2020-03-30', '2020-03-31']

pick = [d.date() for d in data['pick']]
first = [d.date() for d in data['1st_deliver_attempt']]
second = [np.datetime64('1970-01-01') if pd.isnull(d) else d.date() for d in data['2nd_deliver_attempt']]

data["1st_attempt"] = np.busday_count(pick, first, weekmask, holidays)
data["2nd_attempt"] = np.busday_count(first, second, weekmask, holidays)
data["is_late"] = (data["1st_attempt"] > data["sla"]) | (data["2nd_attempt"] > 3)
data["is_late"] = data["is_late"].astype(int)
        
data.head(n=20)
output = pd.DataFrame({'orderid': data['orderid'],
                       'is_late': data['is_late']})
output.to_csv('submission.csv', index=False)