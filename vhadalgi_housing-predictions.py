# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data=pd.read_csv('../input/housing.csv')


# Any results you write to the current directory are saved as output.
##Getting the IV,DV
iv=data.iloc[:,0:9].values
dv=data[['median_house_value']].values

##Check how many missing values are available.
missing_value_count=data.isnull().sum()

#print(missing_value_count)
##Filling NA with mean of the values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(iv[:,0:8])
iv[:,0:8]=imputer.fit_transform(iv[:,0:8])

data['total_bedrooms']=data['total_bedrooms'].replace(np.NaN,data['total_bedrooms'].mean())

missing_value_count=data['total_bedrooms'].isnull().sum()
##Performing One Hot Encoder 
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# lbl=LabelEncoder()
# iv['ocean_proximity']=lbl.fit_transform(iv['ocean_proximity'])
# onehotencoder = OneHotEncoder(categorical_features=[9])
# iv = onehotencoder.fit_transform(iv).toarray()
##Get dummies
iv_dummies=pd.get_dummies(iv[:,8])
iv=pd.DataFrame(iv[:,0:8])

iv=pd.concat([iv,iv_dummies],axis=1)
#print(iv)
##Splitting the data set into Test and Train (80-20)
from sklearn.cross_validation import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)
##Import standard scaler
##Performing Standard Scaler to - remove dominance

from sklearn.preprocessing import StandardScaler
Scale=StandardScaler()

iv_train=Scale.fit_transform(iv_train) ##Fit and Transform
iv_test=Scale.transform(iv_test) ##Only Transform
##Performing Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from math import sqrt
regressor=LinearRegression()
##Fit train
regressor.fit(iv_train,dv_train)
y_pred
#=regressor.predict(iv_test)
#print('Accuracy of LR',MAPE(y_pred,dv_test))

# print(pd.DataFrame(y_pred))
# print(pd.DataFrame(dv_test))
#print(pd.DataFrame(y_pred,np.round(dv_test,decimals=0)))
#performing Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
DTRegressor=DecisionTreeRegressor(random_state=0)
DTRegressor.fit(iv_train,dv_train)

##Predicting output
y_pred_DT=DTRegressor.predict(iv_test)
print('Accuracy of DT',MAPE(y_pred_DT,dv_test))
print('RMSE of DT',rmse(y_pred_DT,dv_test))
print('MAE of DT',mae(y_pred,dv_test))