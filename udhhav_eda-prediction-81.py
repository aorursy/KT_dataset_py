# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import RandomizedSearchCV



pd.set_option('display.max_columns',None)

sns.set_style(style='darkgrid')
train = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')

train.head()
train.describe()
train.info()
train.isnull().sum()
train['Additional_Info'].unique()
train['Duration'].value_counts()
train.dropna(inplace=True)
train.shape
train.columns
train['Day_of_Journey'] = pd.to_datetime(train.Date_of_Journey,format='%d/%m/%Y').dt.day

train['Month_of_Journey'] = pd.to_datetime(train.Date_of_Journey,format='%d/%m/%Y').dt.month
train.head()
train.drop(['Date_of_Journey'],axis =1, inplace=True)
train['Dep_hour'] = pd.to_datetime(train['Dep_Time']).dt.hour

train['Dep_min'] = pd.to_datetime(train['Dep_Time']).dt.minute

train.drop(['Dep_Time'],axis =1, inplace=True)
train.head()
train["Arrival_hour"] = pd.to_datetime(train.Arrival_Time).dt.hour

train["Arrival_min"] = pd.to_datetime(train.Arrival_Time).dt.minute

train.drop(["Arrival_Time"], axis = 1, inplace = True)
train.head()
duration = list(train["Duration"])



for i in range(len(duration)):

    if len(duration[i].split()) != 2:   

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   

        else:

            duration[i] = "0h " + duration[i]           



duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))


train['Duration_hours'] = duration_hours

train['Duration_minutes'] = duration_mins
train.head()
train.drop(['Duration'],axis=1,inplace=True)
train['Airline'].value_counts()
sns.catplot(x='Airline',y='Price',data=train.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)

plt.show()
fig_dims = (25, 15)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x='Airline',y='Price', data = train)
Airline = train[['Airline']]

Airline = pd.get_dummies(Airline,drop_first=True)

Airline.head()
train['Source'].value_counts()
plt.hist(train['Source'])
train['Destination'].value_counts()
plt.hist(train['Destination'],bins=15)

plt.show()
sns.catplot(x='Source',y='Price',data = train.sort_values('Price',ascending=False),kind ='boxen',height=6,aspect=3)
sns.catplot(x='Destination',y='Price',data = train.sort_values('Price',ascending=False),kind ='boxen',height=6,aspect=3)
Source = train[['Source']]

Source = pd.get_dummies(Source,drop_first=True)

Source.head()
Destination = train[['Destination']]

Destination= pd.get_dummies(Destination,drop_first=True)

Destination.head()
train['Route'].head()
train.drop('Route',axis=1, inplace=True)

train.head()
train.drop('Additional_Info',axis=1, inplace=True)
train['Total_Stops'].unique()
train['Total_Stops'].replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

train.head()


data_train = pd.concat([train,Airline,Source,Destination],axis=1)

data_train.head()
data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
data_train.shape
plt.figure(figsize = (18,18))

sns.heatmap(train.corr(), annot = True, cmap = "RdYlGn")

plt.show()
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.regplot(x='Duration_hours',y='Price',data=data_train)
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x='Total_Stops',y='Price',data=data_train,palette='rainbow')
test_data = pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')

test_data.head()
print(test_data.info())



print()

print()



print("Null values :")

print("-"*75)

test_data.dropna(inplace = True)

print(test_data.isnull().sum())



# EDA



# Date_of_Journey

test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day

test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month

test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)



# Dep_Time

test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour

test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute

test_data.drop(["Dep_Time"], axis = 1, inplace = True)



# Arrival_Time

test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour

test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(["Arrival_Time"], axis = 1, inplace = True)



# Duration

duration = list(test_data["Duration"])



for i in range(len(duration)):

    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute

        else:

            duration[i] = "0h " + duration[i]           # Adds 0 hour



duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration



# Adding Duration column to test set

test_data["Duration_hours"] = duration_hours

test_data["Duration_mins"] = duration_mins

test_data.drop(["Duration"], axis = 1, inplace = True)





# Categorical data



print("Airline")

print("-"*75)

print(test_data["Airline"].value_counts())

Airline = pd.get_dummies(test_data["Airline"], drop_first= True)



print()



print("Source")

print("-"*75)

print(test_data["Source"].value_counts())

Source = pd.get_dummies(test_data["Source"], drop_first= True)



print()



print("Destination")

print("-"*75)

print(test_data["Destination"].value_counts())

Destination = pd.get_dummies(test_data["Destination"], drop_first = True)



# Additional_Info contains almost 80% no_info

# Route and Total_Stops are related to each other

test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)



# Replacing Total_Stops

test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)



# Concatenate dataframe --> test_data + Airline + Source + Destination

data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)



data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)



print()

print()



print("Shape of test data : ", data_test.shape)
data_test.head()
data_train.columns
X = data_train.drop(['Price'],axis=1)

X.head()
y = data_train['Price']

y.head()
reg = ExtraTreesRegressor()

reg.fit(X,y)
print(reg.feature_importances_)
plt.figure(figsize=(15,10))

imp_features = pd.Series(reg.feature_importances_, index=X.columns)

imp_features.nlargest(20).plot(kind='barh')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
reg_rf = RandomForestRegressor()

reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
metrics.r2_score(y_test, y_pred)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction = rf_random.predict(X_test)
metrics.r2_score(y_test,prediction)