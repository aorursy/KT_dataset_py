import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas_profiling 

import seaborn as sns

sns.set(style="darkgrid")
pd.set_option('display.max_columns', None)

data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data.head()
dt = data.copy() # save original data as 'dt'
pandas_profiling.ProfileReport(data)
# assign extream values to zero

data['adr'].loc[data['adr']==-6.38] = 0
print (data['adr'].mean()) # find mean

print (data['adr'].std()) # find std
len(data['adr'].loc[data['adr']==0]) # find rows with values = 0
random_adr_list = np.random.randint(101.83112153446453 - 50.5357902855456,

                                    101.83112153446453 + 50.5357902855456,

                                    size = 1960)

# generate random list with the range of mean +/- standard deviation 
data['adr'].loc[data['adr']==0] = random_adr_list # replace 0 with the list we generated
data['agent'].corr(data['is_canceled'])
data['agent'] = data['agent'].fillna(data['agent'].mean()) # fill NaN with mean
pd.isnull(data['agent']).sum()
hist_b = data['babies'].hist()
hist_b = data['children'].hist()
data_kids = data['babies'] + data['children']

data['having_kids'] = [0 if x==0 else 1 for x in data_kids]
data['having_kids'].hist(bins=4)
data = data.drop(['babies','children'],axis = 1)
data['booking_changes_boo'] = [0 if x == 0 else 1 for x in data['booking_changes']]
data['booking_changes_boo'].hist(bins = 3,figsize=(10,5))
data = data.drop(['booking_changes'],axis = 1)
data = data.drop(['company'],axis = 1)
data['days_in_waiting_list'].corr(data['is_canceled'])
data['days_in_waiting_list'].loc[data['days_in_waiting_list']>0].corr(data['is_canceled'])
data.plot.scatter(x='days_in_waiting_list',y='is_canceled')
print ( 'the number of 0 in ''previous_bookings_not_canceled ' 'is', 

    ((data['previous_bookings_not_canceled'].loc[data['previous_bookings_not_canceled'] == 0].count())/len(data))*100,'%')

print ( 'the number of 0 in ''previous_bookings_canceled ' 'is', 

    ((data['previous_cancellations'].loc[data['previous_cancellations'] == 0].count())/len(data))*100,'%')
sns.relplot(x="previous_bookings_not_canceled", y="is_canceled", 

            hue="previous_cancellations", palette="ch:r=-.5,l=.75", data=data);
dt_num = data.select_dtypes(include = ['float64', 'int64'])

dt_num_corr = dt_num.corr()['is_canceled'][:-1]

top_features_list = dt_num_corr[abs(dt_num_corr) > 0.1].sort_values(ascending=False) # no correlation lagger than 0.5

print("There is {} strongly correlated values with is_cancled:\n{}".format(len(top_features_list), top_features_list))
dt_num.hist(figsize=(16, 20))
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
data['cat_lead_time'] = pd.cut(data.lead_time,

                               bins=[0,18,69,160,320,737],

                               labels=['0-18','19-69','70-160','161-320','321+'])
data.groupby(['cat_lead_time'])['is_canceled'].mean().plot.bar()
data.groupby(['previous_cancellations'])['is_canceled'].mean().plot.bar()
data.groupby(['required_car_parking_spaces'])['is_canceled'].mean().plot.bar()
data.groupby(['total_of_special_requests'])['is_canceled'].mean().plot.bar()