#import the required libaries for reading the csv file and for plotting the data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read the data using pandas

data = pd.read_csv('/kaggle/input/pyramid-scheme-profit-or-loss/pyramid_scheme.csv')
# Check for the top 5 rows of data

data.head()
# This provides the basic information the data

data.info()
# this explains the basic stat behind the data

data.describe()
# Remove the column unnamed 

data = data.iloc[:,1:]

data.head()
# paiplot the data to check the linearity

plt.figure(figsize=(12,6))

sns.pairplot(data,kind='scatter')

plt.show()
# Correlation matrix

plt.figure(figsize=(12,6))

cor = data.corr()

sns.heatmap(cor,annot=True)
# oulier detection

sns.boxplot(x=data['cost_price'])
sns.boxplot(x=data['profit_markup'])
sns.boxplot(x=data['depth_of_tree'])
sns.boxplot(data['sales_commission'])
sns.boxplot(data['profit'])
# Hence there are no ouliers cook the data

X = data[['cost_price','profit_markup','depth_of_tree','sales_commission']]

X.head(2)
y = data['profit']

y.head(2)
# Import Linear Regresion model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# Split the data into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)
# Fit the model with the train dataset

lr.fit(X_train,y_train)
# predict the model with test data

y_pred = lr.predict(X_test)
# Coeff and intercept

print('coef:',lr.coef_)

print('Intercept:',lr.intercept_)
# Model evaluation

from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test,y_pred)

rsq = r2_score(y_test,y_pred)
# R square value provides how accurate the model is for the data

print('mean sq error:',mse)

print('r square:',rsq)
# visualising the actual and predicted

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(y_test)+1,1)]

plt.plot(c,y_test,color='b',linestyle='-')

plt.plot(c,y_pred,color='r',linestyle='-')

plt.xlabel('index')

plt.ylabel('Profit')

plt.title('Actual vs Predicted')

plt.show()
# Plot the error value

plt.figure(figsize=(12,6))

c = [i for i in range(1,len(y_test)+1,1)]

plt.plot(c,(y_test-y_pred),color='b',linestyle='-')

#plt.plot(c,y_pred,color='r',linestyle='-')

plt.xlabel('index')

plt.ylabel('Profit')

plt.title('Actual vs Predicted')

plt.show()
# import stat model

import statsmodels.api as sm
# Add constant to the train data

X_train_new = X_train

X_train_new = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit()
lm.params
# This helps to identify the column which are really significant

print(lm.summary())
# Finally the predicted and the actual plot

plt.figure(figsize=(12,6))

plt.plot(y_test,y_pred,color='green',linestyle='-',linewidth=1.5)

plt.show()