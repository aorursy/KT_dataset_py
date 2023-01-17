# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainFileName = '../input/train_trips.csv'
orderFileName = '../input/order_items.csv'
testFileName = '../input/test_trips.csv'
##read raw training data
rawTrainData = pd.read_csv(trainFileName,parse_dates=['shopping_started_at','shopping_ended_at'])
rawTestData = pd.read_csv(testFileName,parse_dates=['shopping_started_at'])
#read order information
orderData = pd.read_csv(orderFileName,index_col = False)
rawTrainData['shopping_trip_time'] = rawTrainData['shopping_ended_at'] - rawTrainData['shopping_started_at']
rawTrainData['shopping_trip_time']=pd.to_timedelta(rawTrainData['shopping_trip_time']).dt.total_seconds()
# merge train data with order data
allTrainData = rawTrainData
allTrainData = allTrainData.merge(orderData,how='left',on='trip_id')
newTrainData = allTrainData.groupby(['trip_id','store_id','department_name','item_id','fulfillment_model','shopping_started_at']).agg({'quantity':np.sum,'shopping_trip_time':np.mean,'shopper_id':np.mean}).reset_index()
byShopperIdAverageShoppingTime = newTrainData.groupby(['shopper_id']).agg({'shopping_trip_time':np.mean}).reset_index()
byStoreIdAverageShoppingTime = newTrainData.groupby(['store_id']).agg({'shopping_trip_time':np.mean}).reset_index()
byTripIdAverageShoppingTime = newTrainData.groupby(['trip_id','store_id','shopper_id','shopping_started_at']).agg({'shopping_trip_time':np.mean,'quantity': np.sum,'department_name': 'nunique'}).reset_index().rename(columns = {'department_name':'num_dept_visited'})
print(byTripIdAverageShoppingTime.head(100))
#learnings from average shopping time by trip id - the orders data mentions quantity but in some cases the quantity is a 
#fraction. Let us include these fractions in our initial model and then omit them to see if there is an improvement in
#performance as quantity does not seem to a normalized measure. The fractions might be weight of the product as pounds.
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
p1 = plt.scatter(byShopperIdAverageShoppingTime['shopper_id'], byShopperIdAverageShoppingTime['shopping_trip_time'], 0.4)
plt.ylabel('average shopping time')
plt.xlabel('shopper_id')
plt.title('average shopping time by shopper id')
plt.show()
p2 = plt.bar(byStoreIdAverageShoppingTime['store_id'], byStoreIdAverageShoppingTime['shopping_trip_time'], 1)
plt.ylabel('average shopping time')
plt.xlabel('store_id')
plt.title('average shopping time by store id')
plt.show()

#average shopping time in some stores is much lower than in other stores
# does fractional quantities - quantities where weights need to be measured impact shopping time? Intuition says it should
# let us see if our intuition is correct.
tempVal = pd.to_numeric(byTripIdAverageShoppingTime['quantity'])
tempVal_rounds = tempVal.round()
tempVal_ints = tempVal[tempVal_rounds == tempVal]
tempVal_floats = tempVal[tempVal_rounds != tempVal]
fractionalIdx =tempVal_floats.index.values
fraction_averageShoppingTime = byTripIdAverageShoppingTime.iloc[fractionalIdx]
nonFraction_averageShoppingTime = byTripIdAverageShoppingTime.iloc[tempVal_ints.index.values]
byStore_fraction_averageShoppingTime = fraction_averageShoppingTime.groupby(['store_id']).agg({'shopping_trip_time':np.mean}).reset_index()
byStore_non_fraction_averageShoppingTime = nonFraction_averageShoppingTime.groupby(['store_id']).agg({'shopping_trip_time':np.mean}).reset_index()

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(byStore_fraction_averageShoppingTime['store_id'], byStore_fraction_averageShoppingTime['shopping_trip_time'], c='b', marker="s", label='Fractional Quantity')
ax1.plot(byStore_non_fraction_averageShoppingTime['store_id'],byStore_non_fraction_averageShoppingTime['shopping_trip_time'], c='r', marker="o", label='Non Fractional Quantity')
plt.legend(loc='lower left');
plt.show()
# from the above plots our intuition that when there are fractional quantities like product weights then the shopping time
# does increase. the data needs to be cleansed and we can build our models for fractional quantities and non fractional quantities
# before combining the predicted shopping duration (a weighted average) for the shopping durations.
finalTrainData = newTrainData.groupby(['trip_id','store_id','department_name']).agg({'quantity':np.sum,'shopping_trip_time':np.mean,'shopper_id':np.mean,'item_id':'nunique'}).reset_index()
finalTrainData = finalTrainData.rename(columns = {'item_id':'num_item_department'})
departmentDf = finalTrainData[['trip_id','department_name']]
uniqueDepartments = finalTrainData['department_name'].unique()
uniqueTripIds = finalTrainData['trip_id'].unique()
pivotTrainData_department = pd.pivot_table(finalTrainData,columns = ['department_name'],values=['quantity'],index='trip_id')
pivotTrainData_department = pivotTrainData_department.fillna(0)
pivotTrainData_numItems = pd.pivot_table(finalTrainData,columns = ['department_name'],values=['num_item_department'],index='trip_id')
pivotTrainData_numItems = pivotTrainData_numItems.fillna(0)
pivotTrainData = pd.concat([pivotTrainData_department,pivotTrainData_numItems],axis = 1)
flat_pivotData = pd.DataFrame(pivotTrainData.to_records())
flat_pivotData.columns = [hdr.replace("('num_item_department', '", "num_item_dept.").replace("')", "") for hdr in flat_pivotData.columns]
flat_pivotData.columns = [hdr.replace("('num_item_department',", "num_item_dept.").replace(")", "") for hdr in flat_pivotData.columns]
flat_pivotData.columns = [hdr.replace("('quantity', '", "quantity.").replace("')", "") for hdr in flat_pivotData.columns]
AllTrainData = pd.concat([byTripIdAverageShoppingTime,flat_pivotData],axis = 1)
AllTrainData = AllTrainData.iloc[:,~AllTrainData.columns.duplicated()]
AllTrainData['day_of_week'] = AllTrainData['shopping_started_at'].dt.dayofweek
AllTrainData['date'] = AllTrainData['shopping_started_at'].dt.date
AllTrainData['hour'] = AllTrainData['shopping_started_at'].dt.hour
collapsedTrainData = AllTrainData[['shopper_id','store_id','date','hour','day_of_week','quantity','shopping_trip_time','num_dept_visited']]
tempDF1 = collapsedTrainData.groupby('hour').agg({'quantity':np.mean,'shopping_trip_time':np.mean}).reset_index()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tempDF1['hour'], tempDF1['quantity'], c='b', label='Average Quantity By hour')
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tempDF1['hour'], tempDF1['shopping_trip_time'], c='b', label='Average Time By hour')
plt.show()
tempDF = AllTrainData[['day_of_week','quantity','shopping_trip_time']]
tempDF1 = tempDF.groupby('day_of_week').agg({'quantity':np.mean,'shopping_trip_time':np.mean}).reset_index()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tempDF1['day_of_week'], tempDF1['quantity'], c='b', label='Average Quantity By Day of week')
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tempDF1['day_of_week'], tempDF1['shopping_trip_time'], c='b', label='Average shopping_trip_time By day of week')
plt.show()
y = AllTrainData['shopping_trip_time'].values
#X = collapsedTrainData[['shopper_id','store_id','hour','day_of_week','quantity','num_dept_visited']]
X = AllTrainData.drop(['shopping_trip_time','trip_id','shopping_started_at','date'],axis=1)
print(X)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.25)
reg = LinearRegression().fit(X_train,y_train)
print('The coefficients for this linear fit are: ',reg.coef_) 
print('The intercept for this linear fit are: ', reg.intercept_) 
print('The score on test data for least squares regression is: ', reg.score(X_val,y_val))
from sklearn.ensemble import GradientBoostingRegressor
GBRegressor = GradientBoostingRegressor(learning_rate = 0.3,n_estimators = 2000,max_depth=3,min_samples_split = 5,loss='ls')
GBRegressor.fit(X_train,y_train)
print('The Gradient Boosting score is: ', GBRegressor.score(X_val,y_val))