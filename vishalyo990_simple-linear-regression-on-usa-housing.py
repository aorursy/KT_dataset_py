#Import the needed packages

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#import the dataset

df  = pd.read_csv('../input/USA_Housing.csv')
#Look at the dataset

df.head(5)
#Looking the data types of dataset

#df.info()
#Looking at mathematical stats for the datasets

df.describe()
df.columns
#Creating the pair-plot for the dataset 

sns.pairplot(df)
#Creating distribution plot for the price

#To see how the price is distributed along the dataset

sns.distplot(df['Price'])
#making a heatmap for the correlation of dataset

fig = plt.figure(figsize = (10,7))

sns.heatmap(df.corr(), annot = True,cmap = "coolwarm")
#Predicting Features

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]
#response feature

y = df['Price']
from sklearn.cross_validation import train_test_split
#Dividing our dataset in train and test data's

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size = 0.4, random_state=101)
from sklearn.linear_model import LinearRegression
#lets fit a model

lm = LinearRegression()
lm.fit(X_train,y_train)
#Intercept for our predictions

print(lm.intercept_)
#Coefficient for our predictions

lm.coef_
#Joining the coefficient with its features

cdf = pd.DataFrame(lm.coef_,X.columns, columns = ['coeff'])
cdf
#predicting the models for test dataset

predictions = lm.predict(X_test)
#Plotting the predictions

plt.scatter(y_test, predictions)
#Residuals

sns.distplot((y_test-predictions))
from sklearn import metrics
#Some mathametics errors

#lesser the error more accurate is our predictions.

#Mean absolute error

metrics.mean_absolute_error(y_test, predictions)
#Mean squared error

metrics.mean_squared_error(y_test, predictions)
#Root mean squared error

np.sqrt(metrics.mean_squared_error(y_test, predictions))