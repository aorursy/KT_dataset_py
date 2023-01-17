#import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Load Data



train_data = pd.read_csv("/kaggle/input/random-linear-regression/train.csv") 

test_data = pd.read_csv("/kaggle/input/random-linear-regression/test.csv")
#Let's look at their shape

print(train_data.shape)

print(test_data.shape)
train_data.info()
test_data.info()
train_data.dropna(inplace = True) 

#I drop missing value since there is one independent variable
from sklearn.linear_model import LinearRegression



# linear regression model with train data

linear_reg = LinearRegression()



x_train = train_data.x.values.reshape(-1, 1)

y_train = train_data.y.values.reshape(-1, 1)



linear_reg.fit(x_train, y_train) #Fitting data



y_train_head = linear_reg.predict(x_train) #Make a prediction with x_train



#Graph of train data and prediction

plt.scatter(x_train, y_train)

plt.plot(x_train, y_train_head, color = "red", linewidth = 2)

plt.title("Train Data")

plt.show()
x_test = test_data.x.values.reshape(-1, 1)

y_test = test_data.y.values.reshape(-1, 1)



# Make a prediction with test data

y_head = linear_reg.predict(x_test)



#Graph of test data and prediction

plt.scatter(x_test, y_test)

plt.plot(x_test, y_head, color = "red", linewidth = 2)

plt.title("Test Data")

plt.show()