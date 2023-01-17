# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/years-of-experience-and-salary-dataset/Salary_Data.csv')

df.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
X,y = np.array(df.YearsExperience).reshape(-1,1), np.array(df.Salary).reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

X_train.shape
model = LinearRegression()

model.fit(X_train, y_train)
#Visualize model with LinearReg using sklearn

plt.scatter(X_train, y_train, color = "red")

plt.plot(X_train, model.predict(X_train), color = "green")

plt.title("Salary vs Experience (Training set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
avg_x = np.average(X_train)

avg_y = np.average(y_train)

avg_x, avg_y

m = 0

n = 0

for x,y in zip(X_train,y_train):

    m = m + (x - avg_x)*(y- avg_y)

    n = n + (x - avg_x)**2

coef = m/n
intercept = np.array([y_train[i] - coef*X_train[i] for i in range(24)]).mean()

coef, intercept
# Coefficient and Intercept of Sklearn model

model.coef_, model.intercept_
Predicted_Value = [coef*x + intercept for x in X_test]

Predicted_Value
model.predict(X_test)
mean_squared_error(y_test, Predicted_Value)
mean_squared_error(y_test, model.predict(X_test))