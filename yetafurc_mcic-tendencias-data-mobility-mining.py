import warnings  

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score



from math import sqrt



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor



from sklearn.externals import joblib



df =  pd.read_csv("../input/taxi-trajectory/train.csv")
df.shape

df.columns

df.head(10)

df.dtypes[df.dtypes == 'object']

df.info()

df.describe()
df.describe(include = ['object'])
df.sort_values('TIMESTAMP',inplace = True)

df['year'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).year) 

df['month'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).month) 

df['month_day'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).day) 

df['hour'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).hour) 

df['week_day'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).weekday()) 
df.head()

plt.figure(figsize = (10,10))

plt.pie(df['year'].value_counts(), labels = df['year'].value_counts().keys(),autopct = '%.1f%%')
plt.figure(figsize = (5,5))

plt.title('Count of trips per day of week')

sns.countplot(y = 'week_day', data = df)

plt.xlabel('Count')

plt.ylabel('Day')
plt.figure(figsize = (10,10))

plt.title('Count of trips per month')

sns.countplot(y = 'month', data = df)

plt.xlabel('Count')

plt.ylabel('Month')
plt.figure(figsize = (10,10))

plt.title('Count of trips per hour')

sns.countplot(y = 'hour', data = df)

plt.xlabel('Count')

plt.ylabel('Hours')
df['MISSING_DATA'].value_counts()

# Eliminación de los datos marcados como "missing data" (datos perdidos) 

df.drop(df[df['MISSING_DATA'] == True].index, inplace = True)

df['MISSING_DATA'].unique()

# Eliminación de los datos cuyo registro "POLYLINE" se considera vacío

df[df['POLYLINE'] =='[]']['POLYLINE'].value_counts()

df.drop(df[df['POLYLINE'] =='[]']['POLYLINE'].index, inplace = True)

df[df['POLYLINE'] =='[]']['POLYLINE'].value_counts()

df['Polyline Length'] = df['POLYLINE'].apply(lambda x : len(eval(x))-1)

df['Trip Time(sec)'] = df['Polyline Length'].apply(lambda x : x * 15)

df.head()
df['Trip Time(sec)'].describe()
df.groupby('week_day').mean()
df['DAY_TYPE'].isnull().sum()
df = pd.get_dummies(df, columns=['CALL_TYPE'])

df.shape

df = df.drop_duplicates()

print(df.shape)

df.to_csv('Cleaned_data.csv', index = None)
df = df.iloc[:50000]

X = df[['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']]

y = df['Trip Time(sec)']

s = StandardScaler()

X = s.fit_transform(X)

print(np.mean(X))

np.std(X)
#### Train and Test splits : 70-30
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3)

print("The size of training input is", X_train.shape)

print("The size of training output is", y_train.shape)

print(50 *'*')

print("The size of testing input is", X_test.shape)

print("The size of testing output is", y_test.shape)
1. ### Machine Learning Models
k_range  =list(range(1,30)) 

param =dict(n_neighbors =k_range)

knn_regressor =GridSearchCV(KNeighborsRegressor(),param,cv =10)

knn_regressor.fit(X_train,y_train)


print(knn_regressor.best_estimator_)

knn_regressor.best_params_


y_train_pred =knn_regressor.predict(X_train) ##Predict train result

y_test_pred =knn_regressor.predict(X_test) ##Predict test result
print("Train Results for KNN Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
depth  =list(range(3,30))

param_grid =dict(max_depth =depth)

tree =GridSearchCV(DecisionTreeRegressor(),param_grid,cv =10)

tree.fit(X_train,y_train)
print(tree.best_estimator_)

tree.best_params_
y_train_pred =tree.predict(X_train) ##Predict train result

y_test_pred =tree.predict(X_test) ##Predict test result
print("Train Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))