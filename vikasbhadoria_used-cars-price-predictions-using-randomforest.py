import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
dataset = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')
dataset.head()
dataset.shape
dataset.info()
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], 
                                                    dataset.iloc[:, -1], 
                                                    test_size = 0.3, 
                                                    random_state = 42)
X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]
X_train['Name'].value_counts()
make_train = X_train['Name'].str.split(' ',expand = True)
make_test = X_test['Name'].str.split(' ',expand = True)

X_train['Manufacturer'] = make_train[0]
X_test["Manufacturer"] = make_test[0]
plt.figure(figsize = (12, 8))
plot = sns.countplot(x = 'Manufacturer', data = X_train)
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')

plt.title("Count of cars based on manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count of cars")
# Let's see the Transmission vs. Price
plt.figure(figsize=(12,6))
sns.boxplot(x='Price',y='Transmission',data=dataset)
plt.title("Price distribution according to the transmission type of the car", fontsize=20,ha='center')
plt.figure(figsize=(14,6))
sns.boxplot(x='Price',y='Location',data=dataset)
plt.title("Price distribution according to the sales location of the car", fontsize=20,ha='center')
axis = dataset.groupby('Year')[['Price']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.title("Prices of the cars as per the year of sales", fontsize=20,ha='center')
import plotly.express as px
axis = dataset.groupby('Year')[['Kilometers_Driven']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.title("Kilometers driven over the course of years", fontsize=20,ha='center')
fig = px.scatter(dataset,x='Price', y='Kilometers_Driven')
fig.update_layout(title='Price v/s Kilometers_driven',xaxis_title="Kilometers Driven",yaxis_title="Price")
fig.show()
fig = px.scatter(dataset,x='Seats', y='Price')
fig.update_layout(title='Price v/s No. of seats',xaxis_title="Number of seats",yaxis_title="Price")
fig.show()
X_train.drop('Name',axis = 1,inplace = True)
X_test.drop('Name', axis = 1, inplace = True)
X_train.drop('Location',axis = 1,inplace = True)
X_test.drop('Location', axis = 1, inplace = True)
curr_time = datetime.datetime.now()
X_train['Year'] = X_train['Year'].apply(lambda x: curr_time.year - x)
X_test['Year'] = X_test['Year'].apply(lambda x: curr_time.year - x)
X_train["Kilometers_Driven"]
mileage_train = X_train["Mileage"].str.split(" ", expand = True)
mileage_test = X_test["Mileage"].str.split(" ", expand = True)

X_train["Mileage"] = pd.to_numeric(mileage_train[0], errors = 'coerce')
X_test["Mileage"] = pd.to_numeric(mileage_test[0], errors = 'coerce')
print(sum(X_train["Mileage"].isnull()))
print(sum(X_test["Mileage"].isnull()))
X_train['Mileage'].fillna(X_train['Mileage'].astype('float64').mean(),inplace = True)
X_test['Mileage'].fillna(X_test['Mileage'].astype('float64').mean(),inplace = True)
cc_train = X_train["Engine"].str.split(" ", expand = True)
cc_test = X_test["Engine"].str.split(" ", expand = True)
X_train["Engine"] = pd.to_numeric(cc_train[0], errors = 'coerce')
X_test["Engine"] = pd.to_numeric(cc_test[0], errors = 'coerce')

bhp_train = X_train["Power"].str.split(" ", expand = True)
bhp_test = X_test["Power"].str.split(" ", expand = True)
X_train["Power"] = pd.to_numeric(bhp_train[0], errors = 'coerce')
X_test["Power"] = pd.to_numeric(bhp_test[0], errors = 'coerce')
X_train["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)
X_test["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)

X_train["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)
X_test["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)

X_train["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)
X_test["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)
X_train.drop(["New_Price"], axis = 1, inplace = True)
X_test.drop(["New_Price"], axis = 1, inplace = True)
X_train = pd.get_dummies(X_train, columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],drop_first = True)
X_test = pd.get_dummies(X_test, columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],drop_first = True)
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]
print(missing_cols)
X_train.shape,X_test.shape
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)
r2_score(y_test, y_pred)
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)
from sklearn.model_selection import RandomizedSearchCV
RF_params={'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
random_search_rf=RandomizedSearchCV(rf,param_distributions=RF_params,n_iter=10,n_jobs=-1,cv=5,verbose=3)
random_search_rf.fit(X_train,y_train)
random_search_rf.best_params_
rf = RandomForestRegressor(n_estimators= 1600,
 min_samples_split= 5,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 50,
 bootstrap= False)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)