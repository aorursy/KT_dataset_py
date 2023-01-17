import pandas as pd
data=pd.read_csv('../input/911.csv')
data.head()
def title_spliter(x):

    x=x.split(':')

    return x[0]
data['type_of_call']=data['title'].apply(title_spliter)
data.head()
data['type_of_call'].unique()
type_of_call=data['type_of_call'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
from decimal import Decimal
plt.figure(figsize=(15, 5))

ax = type_of_call.plot.bar()

for p in ax.patches:

    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))

plt.xticks(rotation=0)

plt.savefig('type_of_calls.png')
data['timeStamp'] = pd.to_datetime(data['timeStamp'], infer_datetime_format=True)
data['timeStamp'].head()
import datetime as dt

data['year'] = data['timeStamp'].dt.year

data['month'] = data['timeStamp'].dt.month_name()

data['day'] = data['timeStamp'].dt.day_name()

data['hour'] = data['timeStamp'].dt.hour
data.head()

data.info()
month_call=data.groupby(['month','type_of_call'])['type_of_call'].count()
month_call
month_call_percentage=month_call.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
month_call_percentage
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_call_percentage=month_call_percentage.reindex(month_order,level=0)
month_call_percentage
month_call_percentage=month_call_percentage.reindex(['EMS','Traffic','Fire'],level=1)
month_call_percentage
sns.set(rc={'figure.figsize':(12, 8)})

month_call_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month')

plt.ylabel('Percentage of Calls')

plt.xticks(rotation=0)

plt.title('Calls/Month')

plt.savefig('month.png')
hours_data = data.groupby(['hour', 'type_of_call'])['type_of_call'].count()
hours_data
hours_call_percentage=hours_data.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
hours_call_percentage
hours_call_percentage=hours_call_percentage.reindex(['EMS','Traffic','Fire'],level=1)
hours_call_percentage
sns.set(rc={'figure.figsize':(12, 8)})

hours_call_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the hour')

plt.ylabel('Percentage of Calls')

plt.xticks(rotation=0)

plt.title('Calls/hours')

plt.savefig('hours.png')
def emergency_separator(x):

    x = x.split(':')

    x = x[1]

    return x



data['emergency_type'] = data['title'].apply(emergency_separator)
data.head()
data['emergency_type'].unique()
emergency_call=data['emergency_type'].value_counts()
emergency_call
data.head()
data1=data[data['type_of_call']=='EMS']
data1.head()
ems_data=data1['emergency_type'].value_counts().head()
data2=data[data['type_of_call']=='Traffic']
traffic_data=data2['emergency_type'].value_counts().head()
data3=data[data['type_of_call']=='Fire']
fire_data=data3['emergency_type'].value_counts().head()
plt.figure(figsize=(10,7))

plt.pie(ems_data.values,labels=ems_data.index,autopct="%.2f")

plt.savefig('EMS_top_5.png')
plt.figure(figsize=(10,7))

plt.pie(traffic_data.values,labels=traffic_data.index,autopct="%.2f")

plt.savefig('Traffic_top_5.png')
plt.figure(figsize=(10,7))

plt.pie(fire_data.values,labels=ems_data.index,autopct="%.2f")

plt.savefig('Fire_top_5.png')