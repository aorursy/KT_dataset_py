# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output
data = pd.read_csv('../input/weatherHistory.csv')
#Error : RMSE
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w
#Pre-processing and Feature Engineering

#Pressure
pressure_median = data['Pressure (millibars)'].median()      
def pressure(x):
    if x==0:
        return x + pressure_median
    else:
        return x
data["Pressure (millibars)"] = data.apply(lambda row:pressure(row["Pressure (millibars)"]) , axis = 1)

#Dropping Loud Cover
data = data.drop('Loud Cover', axis=1)

#Rounding off to 5 decimal points
data['Apparent Temperature (C)'] = round(data['Apparent Temperature (C)'],5)
data['Temperature (C)'] = round(data['Temperature (C)'],5)
data['Visibility (km)'] = round(data['Visibility (km)'],5)

#Formatting Date
data['Date'] = pd.to_datetime(data['Formatted Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['hour'] = data['Date'].dt.hour
data = data.drop(['Formatted Date','Date'],axis=1)


#Function to categorize Summary of weather
def cloud_categorizer(row):
    row = str(row).lower()
    category = ""
    if "foggy" in row:
        category = 5
    elif "overcast" in row:
        category = 4
    elif "mostly cloudy" in row:
        category = 3
    elif "partly cloudy" in row:
        category = 2
    elif "clear" in row:
        category = 1
    else:
        category = 4
    return category 

# Summary Attribute
data["Cloud Summary"] = data.apply (lambda row:cloud_categorizer(row["Summary"]) , axis = 1)
data = data.drop(['Summary'],axis=1)

# Cloud Daily Summary
data["Cloud Daily Summary"] = data.apply (lambda row:cloud_categorizer(row["Daily Summary"]) , axis = 1)
data = data.drop('Daily Summary', axis=1)
data.head()

#Precipitation Type Attribute
data['Precip Type'] = data['Precip Type'].fillna('rain', inplace = False)
le = LabelEncoder()

data.head()
data['Precip Type'] = le.fit_transform(data['Precip Type'])
#Normalization
for i in data.columns:
    if(i=='Precip Type'):
        continue
    mini = min(data[i])
    maxm = max(data[i])
    data[i] = (data[i] - mini)/(maxm - mini)
features = ['Precip Type', 'Temperature (C)','Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)', 'year','month', 'day', 'hour', 'Cloud Summary', 'Cloud Daily Summary']
train, test = train_test_split(data,test_size=0.3,random_state=8)

X_train, X_valid = train_test_split(train, test_size = 0.2)
y_train=X_train['Apparent Temperature (C)']
y_valid=X_valid['Apparent Temperature (C)']

test.to_csv('Test.csv',index=False)
#Training Data on Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X_train[features],y_train)
y_pred = regr.predict(X_valid[features])
error = rmspe(y_valid,y_pred)
error
#Training Data on Decision Tree
regr = DecisionTreeRegressor(max_depth = 6)
regr.fit(X_train[features],y_train)
y_pred = regr.predict(X_valid[features])
error = rmspe(y_valid,y_pred)
error
#Training Data on XGB
regr = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,max_depth = 5, alpha = 10, n_estimators = 500)
regr.fit(X_train[features],y_train)
y_pred = regr.predict(X_valid[features])
error = rmspe(y_valid,y_pred)
error
#Training Data on Random Forest
regr = RandomForestRegressor(n_estimators=500, max_depth=6)
regr.fit(X_train[features],y_train)
y_pred = regr.predict(X_valid[features])
error = rmspe(y_valid,y_pred)
error
"""#Cross Validation using XGB
error = 0
#kf = KFold(n_splits=5)
regr = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,max_depth = 5, alpha = 10, n_estimators = 500)
rkf = RepeatedKFold(n_splits=4, n_repeats=2, random_state=None) 
for train_index, test_index in rkf.split(train):    
    X_train, X_valid = train.iloc[train_index], train.iloc[test_index]
    y_train, y_valid = train['Apparent Temperature (C)'].iloc[train_index], train['Apparent Temperature (C)'].iloc[test_index]
    regr.fit(X_train[features],y_train)
    y_pred = regr.predict(X_valid[features])
    error = error + rmspe(y_valid,y_pred)"""
"""error = error/8
error"""
regr = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,max_depth = 5, alpha = 10, n_estimators = 500)
final = regr.predict(test[features])
print(rmspe(final,test['Apparent Temperature (C)']))
final = final.round(3)
final
#XGB gives us Best RMSE
#Pickling
import pickle
filename = 'xgb_pickle.sav'
pickle.dump(xgb, open(filename,'wb'))