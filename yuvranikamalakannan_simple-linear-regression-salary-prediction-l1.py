import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
# Describe the dataset

df.describe()
# Sample Data

df.head()
df_feature = df[["YearsExperience","Salary"]]
df_feature.head()
# Identify if any null values

df_feature.isnull().sum()
# Identify if any "?" values

df_feature.isin(["?"]).sum()
# Identify if any "? " values

df_feature.isin(["? "]).sum()
viz = df_feature[["Salary","YearsExperience"]]

viz.hist()

plt.show()
plt.scatter(df_feature.Salary, df_feature.YearsExperience,  color='red')

plt.xlabel("Salary")

plt.ylabel("YearsExperience")

plt.show()
df_feature[['YearsExperience','Salary']].corr()['Salary'][:]
# import the libraries

from sklearn import linear_model

from sklearn.model_selection import train_test_split
# Creating dataframe with independent variables only

X = df_feature[['YearsExperience']]
# Creating dataframe with dependent (or) response variable only

Y = df_feature[['Salary']]
# Spliting the test data with 20% and train data with 80%. Keeping the random state with some constant value will ensure to fetch same set of

# train and test records upon multiple executions

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)
lreg = linear_model.LinearRegression()
lreg.fit (X_train, Y_train)
# Finding the coefficients

print ('Coefficients: ', lreg.coef_)

print ('Intercept: ',lreg.intercept_)
type(X_train)
plt.scatter(X_train, Y_train,  color='blue')

plt.plot(X_train, lreg.coef_*X_train + lreg.intercept_, 'r-')

plt.xlabel("Years Of Expereince")

plt.ylabel("Salary")
from sklearn.metrics import r2_score

import math

Y_predict = lreg.predict(X_test)
plt.scatter(X_test, Y_test,  color='blue')

plt.plot(X_test,Y_predict,linestyle='dotted',color='red')

plt.scatter(X_test, Y_predict,  color='red')

plt.xlabel("Years Of Experience")

plt.ylabel("Salary")
MAE = round(np.mean(np.absolute(Y_test - Y_predict)))

MSE = round(np.mean((Y_test - Y_predict)**2))

print("Mean Absolute Error = {}".format(MAE))

print("Mean Square Error = {}".format(MSE))

print("R Square = ",r2_score(Y_predict,Y_test))