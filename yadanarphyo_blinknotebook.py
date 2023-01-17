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
df=pd.read_csv("/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv")
df.head()
# Count unique values in column 'Age' of the dataframe
uniqueShops= df['shopid'].unique().tolist()

import datetime
from datetime import timedelta
hour=1

print(future_date_and_time)
# making boolean series for a team name 

print(filter.shape)
print(filter.head())
filter['event_time'].min()

#for shop in uniqueShops:
    
    
x = df['shopid']==93950878
filter=df[x]
minTime=filter['event_time'].min()
maxTime=filter['event_time'].max()
while future_date_and_time<datetime.datetime.strptime(maxTime, "%Y-%m-%d %H:%M:%S"):
    hours_added = datetime.timedelta(hours = 1)
    future_date_and_time = datetime.datetime.strptime(minTime, "%Y-%m-%d %H:%M:%S") + hours_added
print (filter)
