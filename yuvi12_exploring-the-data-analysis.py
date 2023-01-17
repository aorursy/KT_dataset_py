import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data =pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
data.head()
data.dtypes
data.info()
data.shape
data.isnull().sum()
data['children'] = data['children'].fillna(data['children'].median(),axis=0)
data['country'].value_counts().head(10)
data['country']=data['country'].fillna(value='PRT')
data.drop(['company'],axis=1,inplace=True)

data.drop(['agent'],axis=1,inplace=True)
data.isnull().sum()
sns.countplot(x='hotel',data=data)

plt.title("hotel type")

plt.xlabel("types")

plt.ylabel("no of hotels")

sns.countplot(x='is_canceled',data=data)

plt.title("no of cancellation")

plt.xlabel("cancel yes and no")

plt.ylabel("no of cancellation")
data.dtypes
#outliers

q1 = data.quantile(0.25)

q3 = data.quantile(0.75)

iqr = q3-q1

low = q1-1.5*iqr

up = q3+1.5*iqr

print("lower",str(low))

print("upper",str(up))

print(q1)

print(q3)
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column. 

data['customer_type']= label_encoder.fit_transform(data['customer_type']) 

data['assigned_room_type'] = label_encoder.fit_transform(data['assigned_room_type'])

data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])

data['reservation_status'] = label_encoder.fit_transform(data['reservation_status'])

data['meal'] = label_encoder.fit_transform(data['meal'])

data['country'] = label_encoder.fit_transform(data['country'])

data['distribution_channel'] = label_encoder.fit_transform(data['distribution_channel'])

data['market_segment'] = label_encoder.fit_transform(data['market_segment'])

data['reserved_room_type'] = label_encoder.fit_transform(data['reserved_room_type'])

data['reservation_status_date'] = label_encoder.fit_transform(data['reservation_status_date'])

data['arrival_date_month'] = label_encoder.fit_transform(data['arrival_date_month'])

data['hotel'] = label_encoder.fit_transform(data['hotel'])





print('customer_type:', data['customer_type'].unique())

print('reservation_status', data['reservation_status'].unique())

print('deposit_type', data['deposit_type'].unique())

print('assigned_room_type', data['assigned_room_type'].unique())

print('meal', data['meal'].unique())

print('Country:',data['country'].unique())

print('Dist_Channel:',data['distribution_channel'].unique())

print('Market_seg:', data['market_segment'].unique())

print('reserved_room_type:', data['reserved_room_type'].unique())
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression
sns.countplot(x='previous_cancellations',data=data)

plt.title("previous_cancellations")
x = data.drop(('previous_cancellations'),axis=1)

y = data['previous_cancellations']
#Linear Regression

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)

linear = LinearRegression()

linear.fit(xtrain,ytrain)

pred = linear.predict(xtest)



print("MAE",metrics.mean_absolute_error(ytest,pred))

print("MSE",metrics.mean_squared_error(ytest,pred))

print('RMSE',np.sqrt(metrics.mean_squared_error(ytest, pred)))

print('r2_score:', r2_score(ytest, pred))
#Logistic Regression

logis = LogisticRegression()

logis.fit(xtrain,ytrain)

pred1 = logis.predict(xtest)

print("MAE",metrics.mean_absolute_error(ytest,pred1))

print("MSE",metrics.mean_squared_error(ytest,pred1))

print('RMSE',np.sqrt(metrics.mean_squared_error(ytest, pred1)))

print('r2_score:', r2_score(ytest, pred1))
clf = Lasso(alpha=0.1)



clf.fit(xtrain, ytrain) #training the algorithm



ypred = clf.predict(xtest)



print('Mean Absolute Error_lasso:', metrics.mean_absolute_error(ytest, ypred).round(3))  

print('Mean Squared Error_lasso:', metrics.mean_squared_error(ytest, ypred).round(3))  

print('Root Mean Squared Error_lasso:', np.sqrt(metrics.mean_squared_error(ytest, ypred)).round(3))

print('r2_score_lasso:', r2_score(ytest, ypred).round(3))
ridge = Ridge(alpha=1.0)

ridge.fit(xtrain, ytrain) #training the algorithm



y_pred = ridge.predict(xtest)



print('Mean Absolute Error_ridge:', metrics.mean_absolute_error(ytest, y_pred).round(3))  

print('Mean Squared Error_ridge:', metrics.mean_squared_error(ytest, y_pred).round(3))  

print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(ytest, y_pred)).round(3))

print('r2_score_ridge:', r2_score(ytest, y_pred).round(3))