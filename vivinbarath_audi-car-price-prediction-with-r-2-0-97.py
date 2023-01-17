# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm
data = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
data.head()
data.info()
# There no null values and missing values in the data

data.isna().sum()
# Pairplotting to view the insights

sns.pairplot(data)

plt.show()
# The correlation between the features

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
# Count plot on fuel type

ax = sns.countplot(data.fuelType, label = "Count")
# Count plot on the transmission

ax = sns.countplot(data.transmission, label = "Count")
# Count plot on model

plt.subplots(figsize=(18,18))

ax = sns.countplot(data.model, label = "Count")
# Checking the price of car by transimission type

plt.subplots(figsize=(12,12))

price_by_transmission = data.groupby("transmission")['price'].mean().reset_index()

plt.title("Average Price of vechicle")

sns.set()

sns.barplot(x="transmission", y ="price", data = price_by_transmission)

plt.show()
# Checking the price by fueltype

plt.subplots(figsize=(12,12))

price_by_fuel = data.groupby("fuelType")['price'].mean().reset_index()

plt.title("Average Price of vechicle")

sns.set()

sns.barplot(x="fuelType", y ="price", data = price_by_fuel)

plt.show()
# Checking the price by model

plt.subplots(figsize=(18,18))

price_by_model = data.groupby("model")['price'].mean().reset_index()

plt.title("Average Price of vechicle")

sns.set()

sns.barplot(x="model", y ="price", data = price_by_model)

plt.show()
# Feature engineering

final_data = pd.concat([data,pd.get_dummies(data.fuelType), pd.get_dummies(data.transmission)], axis =1)
final_data.head()
final_data = final_data.drop(["transmission", "fuelType", "model"], axis = 1)
# Fitting Regression Model

X = final_data.drop("price", axis = 1)

y = final_data["price"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)
X_train_new = X_train

X_train_new = sm.add_constant(X_train_new)
lr_1 = sm.OLS(y_train,X_train_new).fit()

print(lr_1.summary())
# The summary helps to view the p-value and decide on the variables to stay on the data
from sklearn import preprocessing

def normalize(x):

    return ((x-np.min(x))/(max(x)-min(x)))



final_data = final_data.apply(normalize)
# After Normalizing

final_data.head()
X = final_data.drop('price',axis=1)

y =  final_data['price']
# Splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)

forest_X_train = X_train.copy()

forest_X_test = X_test.copy()

forest_y_train = y_train.copy()

forest_y_test = y_test.copy()
lr = LinearRegression()

lr.fit(X_train,y_train)
X_train_new = X_train

X_train_new = sm.add_constant(X_train_new)

lr_2 = sm.OLS(y_train,X_train_new).fit()

print(lr_2.summary())
plt.figure(figsize=(12,6))

sns.heatmap(final_data.corr(),annot=True)

plt.show()
# UDF for calculating vif value

def vif_cal(input_data, dependent_col):

    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared  

        vif=round(1/(1-rsq),2)

        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)
vif_cal(input_data=final_data,dependent_col='price')
# dropping automatic column because p-value is high and also VIF is high too

X_train = X_train.drop('Automatic',axis=1)

lr_3 = sm.OLS(y_train,X_train).fit()

print(lr_3.summary())
vif_cal(input_data=final_data.drop(['Automatic'],axis=1),dependent_col='price')
# dropping Diesel because p-value is high and also VIF is high too

X_train = X_train.drop('Diesel',1)

lr_4 = sm.OLS(y_train,X_train).fit()



print(lr_4.summary())
vif_cal(input_data=final_data.drop(['Automatic','Diesel'],axis=1),dependent_col='price')
# Making predictions

X_test_m4 = sm.add_constant(X_test)
X_test_m4 = X_test.drop(['Automatic', 'Diesel'],axis=1)

y_pred_m4 = lr_4.predict(X_test_m4)
from sklearn.metrics import r2_score,mean_squared_error

print('R square:',r2_score(y_test,y_pred_m4))

print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred_m4)))
plt.figure(figsize=(12,6))

c = [i for i in range(1,len(X_test_m4)+1,1)]

plt.plot(c,y_test,linestyle='-',color='b')

plt.plot(c,y_pred_m4,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Price')

plt.show()
# Error

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(X_test_m4)+1,1)]

plt.plot(c,y_test-y_pred_m4,linestyle='-',color='b')

#plt.plot(c,y_pred_m6,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Error')

plt.show()
# Error Distribution

plt.figure(figsize=(12,6))

sns.distplot(y_test-y_pred_m4,bins=50)

plt.xlabel('y_test - y_pred')

plt.ylabel('Index')

plt.title('Error distribution')

plt.show()
# Random forest regressor

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(forest_X_train,forest_y_train)
forest_y_pred = forest.predict(forest_X_test)
# Calculating RMSE

forest_rmse = np.sqrt(mean_squared_error(forest_y_test,forest_y_pred))

forest_r2score = r2_score(forest_y_test,forest_y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )
# Model evaluation

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(forest_X_test)+1,1)]

plt.plot(c,forest_y_test,linestyle='-',color='b')

plt.plot(c,forest_y_pred,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Price')

plt.show()
# Error

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(forest_X_test)+1,1)]

plt.plot(c,forest_y_test-forest_y_pred,linestyle='-',color='b')

#plt.plot(c,y_pred_m6,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Error')

plt.show()
# Decision tree regressor
from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor()

tree.fit(forest_X_train.drop(['Automatic','Diesel'], axis=1),forest_y_train)
tree_y_pred = tree.predict(forest_X_test.drop(['Automatic', 'Diesel'], axis = 1))
# Calculating RMSE

forest_rmse = np.sqrt(mean_squared_error(forest_y_test,tree_y_pred))

forest_r2score = r2_score(forest_y_test,tree_y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )
# XGboost model
import xgboost as xgb

regressor = xgb.XGBRegressor(

    n_estimators=200,

    reg_lambda=2,

    gamma=0,

    max_depth=5

)
regressor.fit(forest_X_train, forest_y_train)
boost_y_pred = regressor.predict(forest_X_test)
# Calculating RMSE

forest_rmse = np.sqrt(mean_squared_error(forest_y_test,boost_y_pred))

forest_r2score = r2_score(forest_y_test,boost_y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )
# Model evaluation

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(forest_X_test)+1,1)]

plt.plot(c,forest_y_test,linestyle='-',color='b')

plt.plot(c,boost_y_pred,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Price')

plt.show()
# Error

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(forest_X_test)+1,1)]

plt.plot(c,forest_y_test-boost_y_pred,linestyle='-',color='b')

#plt.plot(c,y_pred_m6,linestyle='-',color='r')

plt.title('Actual Vs Prediction')

plt.xlabel('Index')

plt.ylabel('Error')

plt.show()
# Error distribution

plt.figure(figsize=(12,6))

sns.distplot(forest_y_test-boost_y_pred,bins=50)

plt.xlabel('y_test - y_pred')

plt.ylabel('Index')

plt.title('Error distribution')

plt.show()
# The prediction

print('The original price'+str(forest_y_test.head())+'\nThe predicted values'+str(boost_y_pred[0:5]))