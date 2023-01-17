# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')

test_data=pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Test_set.xlsx')
train_data.shape,test_data.shape
train_data.head()
train_data.info()
test_data.info()
# Checking missing value in dataset

train_data.isnull().values.any(),test_data.isnull().values.any()
train_data.isnull().sum()
train_data.dropna(inplace=True)
# Checking if there are any Duplicate values

train_data[train_data.duplicated()]
# Drop duplicates value

train_data.drop_duplicates(keep='first',inplace=True)
train_data["Additional_Info"].value_counts()
train_data["Additional_Info"] = train_data["Additional_Info"].replace({'No Info': 'No info'})
# Duration convert hours in min.

train_data['Duration']=  train_data['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

test_data['Duration']=  test_data['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
# Date_of_Journey

train_data["Journey_day"] = train_data['Date_of_Journey'].str.split('/').str[0].astype(int)

train_data["Journey_month"] = train_data['Date_of_Journey'].str.split('/').str[1].astype(int)

train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)



# Dep_Time

train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute

train_data.drop(["Dep_Time"], axis = 1, inplace = True)



# Arrival_Time

train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour

train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

train_data.drop(["Arrival_Time"], axis = 1, inplace = True)
# Date_of_Journey

test_data["Journey_day"] = test_data['Date_of_Journey'].str.split('/').str[0].astype(int)

test_data["Journey_month"] = test_data['Date_of_Journey'].str.split('/').str[1].astype(int)

test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)



# Dep_Time

test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour

test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute

test_data.drop(["Dep_Time"], axis = 1, inplace = True)



# Arrival_Time

test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour

test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(["Arrival_Time"], axis = 1, inplace = True)
plt.figure(figsize = (15, 10))

plt.title('Count of flights month wise')

ax=sns.countplot(x = 'Journey_month', data = train_data)

plt.xlabel('Month')

plt.ylabel('Count of flights')

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')
# Total_Stops

train_data['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)

test_data['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)
train_data["Airline"].value_counts()
plt.figure(figsize = (15, 10))

plt.title('Count of flights with different Airlines')

ax=sns.countplot(x = 'Airline', data =train_data)

plt.xlabel('Airline')

plt.ylabel('Count of flights')

plt.xticks(rotation = 90)

for p in ax.patches:

    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')
plt.figure(figsize = (15, 10))

plt.title('Price VS Airlines')

plt.scatter(train_data['Airline'], train_data['Price'])

plt.xticks(rotation = 90)

plt.xlabel('Airline')

plt.ylabel('Price of ticket')

plt.xticks(rotation = 90)

# Airline

train_data["Airline"].replace({'Multiple carriers Premium economy':'Other', 

                                                        'Jet Airways Business':'Other',

                                                        'Vistara Premium economy':'Other',

                                                        'Trujet':'Other'

                                                   },    

                                        inplace=True)



test_data["Airline"].replace({'Multiple carriers Premium economy':'Other', 

                                                        'Jet Airways Business':'Other',

                                                        'Vistara Premium economy':'Other',

                                                        'Trujet':'Other'

                                                   },    

                                        inplace=True)


plt.figure(figsize = (15, 10))

plt.title('Price VS Additional Information')

sns.scatterplot(train_data['Additional_Info'], train_data['Price'],data=train_data)

plt.xticks(rotation = 90)

plt.xlabel('Information')

plt.ylabel('Price of ticket')
train_data["Additional_Info"].value_counts()
# Additional_Info

train_data["Additional_Info"].replace({'Change airports':'Other', 

                                                        'Business class':'Other',

                                                        '1 Short layover':'Other',

                                                        'Red-eye flight':'Other',

                                                        '2 Long layover':'Other',   

                                                   },    

                                        inplace=True)

test_data["Additional_Info"].replace({'Change airports':'Other', 

                                                        'Business class':'Other',

                                                        '1 Short layover':'Other',

                                                        'Red-eye flight':'Other',

                                                        '2 Long layover':'Other',   

                                                   },    

                                        inplace=True)
train_data.head()
data = train_data.drop(["Price"], axis=1)
train_categorical_data = data.select_dtypes(exclude=['int64', 'float','int32'])

train_numerical_data = data.select_dtypes(include=['int64', 'float','int32'])



test_categorical_data = test_data.select_dtypes(exclude=['int64', 'float','int32','int32'])

test_numerical_data  = test_data.select_dtypes(include=['int64', 'float','int32'])
train_categorical_data.head()
#Label encode and hot encode categorical columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_categorical_data = train_categorical_data.apply(LabelEncoder().fit_transform)

test_categorical_data = test_categorical_data.apply(LabelEncoder().fit_transform)
train_categorical_data.head()
X = pd.concat([train_categorical_data, train_numerical_data], axis=1)

y=train_data['Price']

test_set = pd.concat([test_categorical_data, test_numerical_data], axis=1)
X.head()
y.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score



from math import sqrt



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import KFold



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# training testing and splitting the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print("The size of training input is", X_train.shape)

print("The size of training output is", y_train.shape)

print(50 *'*')

print("The size of testing input is", X_test.shape)

print("The size of testing output is", y_test.shape)
params ={'alpha' :[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

ridge_regressor =GridSearchCV(Ridge(), params ,cv =5,scoring = 'neg_mean_absolute_error', n_jobs =-1)

ridge_regressor.fit(X_train ,y_train)
y_train_pred =ridge_regressor.predict(X_train) ##Predict train result

y_test_pred =ridge_regressor.predict(X_test) ##Predict test result
print("Train Results for Ridge Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Ridge Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))

params ={'alpha' :[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

lasso_regressor =GridSearchCV(Lasso(), params ,cv =15,scoring = 'neg_mean_absolute_error', n_jobs =-1)

lasso_regressor.fit(X_train ,y_train)
y_train_pred =lasso_regressor.predict(X_train) ##Predict train result

y_test_pred =lasso_regressor.predict(X_test) ##Predict test result
print("Train Results for Lasso Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Lasso Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
k_range = list(range(1, 30))

params = dict(n_neighbors = k_range)

knn_regressor = GridSearchCV(KNeighborsRegressor(), params, cv =10, scoring = 'neg_mean_squared_error')

knn_regressor.fit(X_train, y_train)

y_train_pred =knn_regressor.predict(X_train) ##Predict train result

y_test_pred =knn_regressor.predict(X_test) ##Predict test result
print("Train Results for KNN Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))



print("Test Results for KNN Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % errorr: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
depth  =list(range(3,30))

param_grid =dict(max_depth =depth)

tree =GridSearchCV(DecisionTreeRegressor(),param_grid,cv =10)

tree.fit(X_train,y_train)
y_train_pred =tree.predict(X_train) ##Predict train result

y_test_pred =tree.predict(X_test) ##Predict test result
print("Train Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 20, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)

random_regressor.fit(X_train, y_train)

y_train_pred = random_regressor.predict(X_train)

y_test_pred = random_regressor.predict(X_test)
print("Train Results for Random Forest Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Random Forest Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
tuned_params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}

model = RandomizedSearchCV(XGBRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
print("Train Results for XGBoost Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for XGBoost Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("Mean absolute % error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))