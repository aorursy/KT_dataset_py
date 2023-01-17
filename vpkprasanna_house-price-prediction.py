# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/usa-housing-listings/housing.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()
missing = data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
import scipy.stats as st

y = data['price']

plt.figure(1)

plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2) 

plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3) 

plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
data.hist(figsize=(30,10))

correleation_matrix = data.corr()
plt.figure(figsize=(20,20))

sns.heatmap(correleation_matrix, cbar=True, square= True,fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')

plt.figure(figsize=(20,20))



sns.countplot(data['baths'])
plt.figure(figsize=(20,20))

sns.countplot(data['laundry_options'])
plt.figure(figsize=(20,20))

sns.countplot(data['parking_options'])
plt.figure(figsize=(20,20))

sns.countplot(data['comes_furnished'])
plt.figure(figsize=(20,20))

sns.countplot(data['beds'])
data['laundry_options'] = data['laundry_options'].fillna(data['laundry_options'].mode()[0])

data['parking_options'] = data['parking_options'].fillna(data['parking_options'].mode()[0])

data['state'] = data['state'].fillna(data['state'].mode()[0])

data['lat'] = data['lat'].fillna(data['lat'].mean())

data['long'] = data['long'].fillna(data['long'].mean())
data = data.drop(['url','region_url','image_url','description','state'],axis=1)
le = LabelEncoder()

data['region'] = le.fit_transform(data['region'])

data['laundry_options'] = le.fit_transform(data['laundry_options'])

data['parking_options'] = le.fit_transform(data['parking_options'])

data['type'] = le.fit_transform(data['type'])
y = data['price']

X = data.drop(['price'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

# Import library for Linear Regression

from sklearn.linear_model import LinearRegression



# Create a Linear regressor

lm = LinearRegression()



# Train the model using the training sets 

lm.fit(X_train, y_train)
lm.intercept_
#Converting the coefficient values to a dataframe

coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T

coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})

coeffcients
y_pred = lm.predict(X_test)
from sklearn import metrics

import numpy as np

# Model Evaluation

print('R^2:',metrics.r2_score(y_test, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Import XGBoost Regressor

from xgboost import XGBRegressor



#Create a XGBoost Regressor

reg = XGBRegressor()



# Train the model using the training sets 

reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)

acc_xgb = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_xgb)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
# # Creating scaled set to be used in model to improve our results

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)
# # Import SVM Regressor

# from sklearn import svm



# # Create a SVM Regressor

# reg = svm.SVR()
# reg.fit(X_train, y_train)

# y_test_pred = reg.predict(X_test)

# # Model Evaluation

# acc_svm = metrics.r2_score(y_test, y_test_pred)

# print('R^2:', acc_svm)

# print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

# print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

# print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

# print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100)
gbr.fit(X_train,y_train)
y_test_pred = gbr.predict(X_test)

acc_xgb = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_xgb)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))