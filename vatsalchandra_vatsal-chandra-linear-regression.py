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
# input dataset

data = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")

data.head()      #shows first 5 rows of the dataset
import matplotlib.pyplot as plt

plt.scatter(data["YearsExperience"], data["Salary"])
#Plotting the Correlation Matrix

import seaborn as sn

print(data.corr())

sn.heatmap(data.corr(), annot=True)
# MSE

def MSE(actual, predicted):

    return  (((actual - predicted) ** 2).mean())



#RMSE

def RMSE(MSE):

    return np.sqrt(MSE)

x = data.iloc[:, :-1]   # YearsExperience = x

y = data.iloc[:, [-1]]  # Salary = y



#splitting in 50% train set and 50%test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.50,test_size = 0.50) 



#training

from sklearn import linear_model

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

y_predicted = regression.predict(x_test)



print("Using my own functions:")

print("mse= " , MSE(y_test,y_predicted))

print("rmse= " , RMSE(MSE(y_test, y_predicted)))
from sklearn.metrics import mean_squared_error

mse_sk= mean_squared_error(y_test, y_predicted)

print("Using sklearn matrices:")

print("mse= ", mse_sk)

print("rmse= ", RMSE(mse_sk))
plt.scatter(x_test, y_test,  color='red')

plt.plot(x_test, regression.predict(x_test))
x = data.iloc[:, :-1]   # YearsExperience = x

y = data.iloc[:, [-1]]  # Salary = y



#splitting in 70% train set and 30%test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.70,test_size = 0.30) 



#training

from sklearn import linear_model

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

y_predicted = regression.predict(x_test)

print("Using my own functions:")

print("mse= " , MSE(y_test,y_predicted))

print("rmse= " , RMSE(MSE(y_test, y_predicted)))
from sklearn.metrics import mean_squared_error

mse_sk= mean_squared_error(y_test, y_predicted)

print("Using sklearn matrices:")

print("mse= ", mse_sk)

print("rmse= ", RMSE(mse_sk))
plt.scatter(x_test, y_test,  color='red')

plt.plot(x_test, regression.predict(x_test))
x = data.iloc[:, :-1]   # YearsExperience = x

y = data.iloc[:, [-1]]  # Salary = y



#splitting in 80% train set and 20%test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.80,test_size = 0.20) 



#training

from sklearn import linear_model

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

y_predicted = regression.predict(x_test)
print("Using my own functions:")

print("mse= " , MSE(y_test,y_predicted))

print("rmse= " , RMSE(MSE(y_test, y_predicted)))
from sklearn.metrics import mean_squared_error

mse_sk= mean_squared_error(y_test, y_predicted)

print("Using sklearn matrices:")

print("mse= ", mse_sk)

print("rmse= ", RMSE(mse_sk))
plt.scatter(x_test, y_test,  color='red')

plt.plot(x_test, regression.predict(x_test))