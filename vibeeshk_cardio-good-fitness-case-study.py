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
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
filepath= '/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv'



df = pd.read_csv(filepath)

df.head()
#Change the categorical variables to numbers

df['Gender'] = df['Gender'].replace('Male', 0)

df['Gender'] = df['Gender'].replace('Female', 1)

df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 0)

df['MaritalStatus'] = df['MaritalStatus'].replace('Partnered', 1)

df.head()
#Perform one hot encoding on the Products column

one_hot = pd.get_dummies(df['Product'])

# Drop column Product as it is now encoded

df = df.drop('Product',axis = 1)

# Join the encoded df

df = df.join(one_hot)

df.head()
#Defining the independant and depandant variables

from sklearn.model_selection import train_test_split

y=df['Fitness']

x=df.drop('Fitness',axis=1)

#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
##Now we will run a few machine learning techiniques to see which one is the most applicable



#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x_train, y_train)

y_predicted = linearRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
#Polynomial Regression

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(x_train)

x_poly_test = polynomial_features.fit_transform(x_test)

model = LinearRegression()

model.fit(x_poly, y_train)

y_predicted_p = model.predict(x_poly_test)

mse = mean_squared_error(y_test, y_predicted_p)

r = r2_score(y_test, y_predicted_p)

mae = mean_absolute_error(y_test,y_predicted_p)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# Decision Tree - CART

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(x_train, y_train)

y_predicted_d = regressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_d)

r = r2_score(y_test, y_predicted_d)

mae = mean_absolute_error(y_test,y_predicted_d)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# Random Forest

rf = RandomForestClassifier()

rf.fit(x_train,y_train);

y_predicted_r = rf.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_r)

r = r2_score(y_test, y_predicted_r)

mae = mean_absolute_error(y_test,y_predicted_r)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)