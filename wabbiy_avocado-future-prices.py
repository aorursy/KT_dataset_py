import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats
# Upload and view the dataset

data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data.head(8)
data.shape
# All the regions (cities)

data.region.unique()

# The type of avocados

data.type.unique()
# What are the Data type and do we have some null

data.info()
# do we have any duplicates, that we may need to drop

data.duplicated().sum()
data.describe()
# compare the prices of the two types and identify any outliers



#define the plot

f,ax = plt.subplots(figsize = (10,7))



sns.boxplot(x="type", y="AveragePrice",data=data,);

plt.title("Average Price Per Piece",fontsize = 25,color='black')

plt.xlabel('Type of Avocado',fontsize = 15,color='black')

plt.ylabel('Avg Price',fontsize = 15,color='black')
# compare the price over the four years and identify any outliers

f,ax = plt.subplots(figsize = (10,7))

sns.boxplot(x="year", y="AveragePrice",data=data,);

plt.title("Average Price Per Piece",fontsize = 25,color='black')

plt.xlabel('year',fontsize = 15,color='black')

plt.ylabel('Avg Price',fontsize = 15,color='black')
# plot AveragePrice distribution

sns.distplot(data['AveragePrice']);
# what is the relationship between the price and the rest of the features

# Is there a correlation

# Perform correlation



corr = data.corr()

corr.sort_values(["AveragePrice"], ascending = False, inplace = True)

print(corr.AveragePrice)


data.shape
# convert Date from object to a datetime

import datetime as dt

data.Date= pd.to_datetime(data.Date) 

data.Date.dt.month

data["Month"]= data.Date.dt.month
data.info()
#drop unecessary features

data.drop(['Date'],axis =1,inplace=True)

data.drop(['Unnamed: 0'],axis =1,inplace=True)

data.drop(['year'],axis =1,inplace=True)
#get dummies for categorical data

data = pd.get_dummies(data)
# checking the shape at this point

data.shape
#Define X and y value.

x=data.drop('AveragePrice',axis=1)

y=data ['AveragePrice']



from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict



x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
#Use numpy to convert to array

x_train = np.array(x_train)

y_train = np.array(y_train)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(x_train, y_train);
#Use numpy to convert to test train to an array

x_test = np.array(x_test)

y_test = np.array(y_test)
# Use the forest's predict method on the test data

predictions = rf.predict(x_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
