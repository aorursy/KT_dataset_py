import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/salaries-data/Salary_Data.csv')
dataset.head()
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
print(X)
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Regression is when you to predict the continous value like Salary

#Classification is when you have to predict a Categoty

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test Set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()