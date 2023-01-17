import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



sns.set()
train = pd.read_csv('../input/training.csv').set_index('RefId')

test = pd.read_csv('../input/test.csv').set_index('RefId')



train['kind'] = 'train'

test['kind'] = 'test'



dataset = train.copy()

dataset.head()
print("dataCount:",len(dataset))

print('Nan in data:\n',dataset.isnull().sum())
print(train.groupby("IsBadBuy").size())
corr = dataset.corr()

corr.style.background_gradient(cmap='coolwarm')
grid = sns.FacetGrid(dataset, col="IsBadBuy",hue = 'IsBadBuy' ,height=5)

grid.map(sns.distplot,'VehBCost',bins = 30);

grid.fig.suptitle('cost paid for the vehicle vs IsBadBuy')

grid.fig.set_size_inches(15,8)

plt.show()



grid = sns.FacetGrid(dataset, col="IsBadBuy",hue = 'IsBadBuy',height=5)

grid.map(sns.distplot,'VNZIP1',bins = 30);

grid.fig.suptitle('buy location zipcode vs IsBadBuy')

grid.fig.set_size_inches(15,8)

plt.show()
dataset['RoundVehBCost'] = round(dataset['VehBCost'],-2)

dataset.groupby('RoundVehBCost').agg([np.mean,np.size])['IsBadBuy'].query('size > 100')['mean'].plot(figsize=(14,5), title = "RoundVehBCost Vs IsBadBuy")

plt.show()
dataset.groupby('VehicleAge').agg([np.mean,np.size])['IsBadBuy'].query('size > 100')['mean'].plot(title = "VehicleAge Vs IsBadBuy")

plt.show()
dataset.groupby("VehYear").mean()["IsBadBuy"].plot.bar(title = "VehYear Vs IsBadBuy")

plt.show()
dataset.groupby('Color').mean()['IsBadBuy'].plot.bar(title = "Color Vs IsBadBuy")

plt.show()



dataset['Color'].dropna(inplace = True)

train['Color'].dropna(inplace = True)


dataset.groupby('Make').agg([np.mean,np.size])['IsBadBuy'].query('size > 100')['mean'].plot.bar(figsize=(14,5), title = "Make Vs IsBadBuy")

plt.show()
dataset.groupby('WheelType').mean()['IsBadBuy'].plot.bar(title = "IsBadBuy Vs WheelType")

plt.show()
wheel_groupby_multi_index = dataset.groupby(['WheelTypeID','WheelType']).sum().index

print(pd.DataFrame(wheel_groupby_multi_index.get_level_values(1), index = wheel_groupby_multi_index.get_level_values(0)))



print("\nWheelTypeID and WheelType is the same param")



#WheelTypeID and WheelType is the same param

#drop WheelTypeID and use only   WheelType

dataset.drop('WheelTypeID',axis = 1,inplace = True)
dataset.groupby("IsOnlineSale").mean()['IsBadBuy'].plot.bar()

plt.show()
dataset.groupby('VNST').agg([np.mean,np.size])['IsBadBuy'].query('size > 100')['mean'].plot.bar(figsize=(14,5), title = "VNST Vs IsBadBuy")

plt.show()
dataset.groupby('BYRNO').agg([np.mean,np.size])['IsBadBuy'].query('size > 10')['mean'].plot.bar(figsize=(14,5), title = "BYRNO Vs IsBadBuy")

plt.show()
# dataset['launched_hour'] = pd.DatetimeIndex(dataset['PurchDate']).hour

# dataset['dayfloat']=pd.DatetimeIndex(dataset['PurchDate']).day+dataset.launched_hour/24.0

dataset['monthfloat']=pd.DatetimeIndex(dataset['PurchDate']).month # +dataset.dayfloat



dataset['x_purch_date']=np.sin(2.*np.pi*dataset.monthfloat/12.)

dataset['y_purch_date']=np.cos(2.*np.pi*dataset.monthfloat/12.)



ax = sns.scatterplot(x="x_purch_date", y="y_purch_date", hue="IsBadBuy",style="IsBadBuy",alpha = 0.4,palette = 'Set1_r', data=dataset)

ax.set_title("PurchDate")
dataset.groupby('Nationality').agg([np.mean,np.size])['IsBadBuy'].query('size > 10')['mean'].plot.bar( title = "Nationality Vs IsBadBuy")

plt.show()