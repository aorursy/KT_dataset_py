# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/911.csv')
data.head()
def call_type_separator(x):

    x=x.split(':')

    return x[0]
data['call_type']=data['title'].apply(call_type_separator)
data.head()
data.call_type.unique()
call_types=data.call_type.value_counts()
call_types
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))

call_types.plot.bar()
data.timeStamp=pd.to_datetime(data['timeStamp'], infer_datetime_format=True)
data.info()
import datetime as dt
data['year']=data['timeStamp'].dt.year

data['month']=data['timeStamp'].dt.month_name()

data['day']=data['timeStamp'].dt.day_name()

data['hour']=data['timeStamp'].dt.hour
data.head()
def emergency_type_separator(x):

    x=x.split(':')

    return x[1]
data['emergency_type']=data['title'].apply(emergency_type_separator)
data.head()
calls_data=data.groupby(['month','call_type'])['call_type'].count()
calls_data.head()
calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
calls_data_percentage.head()
font = {

    'size': 'x-large',

    'weight': 'bold'

}
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)
calls_data_percentage.head()
calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)
calls_data_percentage.head()
import seaborn as sns
sns.set(rc={'figure.figsize':(12, 8)})

calls_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)
hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()
hours_data.head()
hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
hours_data_percentage.head()
hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)
hours_data_percentage.head()
sns.set(rc={'figure.figsize':(18, 8)})

hours_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Hour of the day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Hour', fontdict=font)
ems_data = data[data['call_type'] == 'EMS']['emergency_type'].value_counts()[:8]
plt.pie(ems_data, labels=ems_data.index, autopct="%.2f")

plt.title('EMS-DATA',fontdict=font)
fire_data=data[data['call_type'] == 'Fire']['emergency_type'].value_counts()[:8]
plt.pie(fire_data, labels=fire_data.index, autopct="%.2f")

plt.title('FIRE-DATA', fontdict=font)
traffic_data=data[data['call_type'] == 'Traffic']['emergency_type'].value_counts()[:5]
plt.pie(traffic_data, labels=traffic_data.index, autopct="%.2f")

plt.title('TRAFFIC-DATA', fontdict=font)