import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/Salary_Data.csv")
dataset.head()
x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1].values
dataset.info()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color= 'red')

plt.plot(x_train,regressor.predict(x_train),color = 'blue')

plt.xlabel("Years of Experience")

plt.ylabel("Salary")
plt.scatter(x_test,y_test,color= 'red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.xlabel("Years of Experience")

plt.ylabel("Salary")