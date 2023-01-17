import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data_path = '/kaggle/input/ex1data2.txt'



data = pd.read_csv(data_path, sep = ',', header = None)

data.columns = ['Living Area', 'Bedrooms', 'Price']







# Print out first 5 rows to get the imagination of data

data.head()



# Is there any missing data or not?

data.isnull().values.any()
# Plot histogram of each coumn

#1. Histogram of Bedrooms

data[['Bedrooms']].plot(kind = 'hist', bins = [0, 1, 2, 3, 4, 5, 6], rwidth = 0.8)
# Plot histogram of each coumn

#2. Histogram of Living Area

data[['Living Area']].plot(kind = 'hist', bins = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], rwidth = 0.5)
# Plot histogram of each coumn

#3. Histogram of Price

data[['Price']].plot(kind = 'hist', bins = [200000, 250000, 300000, 350000, 400000, 450000], rwidth = 0.8)
# Plot 2 columns as a scatter plot

#1. Number of Bedrooms vs Price

data.plot(kind='scatter', x = 'Bedrooms', y = 'Price', color = 'green')
# Plot 2 columns as a scatter plot

#2. Living Area vs Number of Bedrooms

data.plot(kind = 'scatter', x = 'Living Area', y = 'Bedrooms', color = 'red')
# Plot 2 columns as a scatter plot

#3. Living Area vs Price

data.plot(kind='scatter', x = 'Living Area', y = 'Price', color = 'blue')
import matplotlib.pyplot as plt

import math

from sklearn.linear_model import LinearRegression as lm
# Predict Price from Bedrooms

X = data['Bedrooms'].values.reshape(-1, 1)

Y = data['Price'].values.reshape(-1, 1)



# Visualize the data

plt.scatter(X, Y)



# Train the model

model=lm().fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression model

plt.plot(X, Y_pred, color = 'red')



plt.show()
# Predict Price from Living Area

X = data['Living Area'].values.reshape(-1, 1)

Y = data['Price'].values.reshape(-1, 1)



# Visualize the data

plt.scatter(X, Y)



# Train the model

model=lm().fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression model

plt.plot(X, Y_pred, color = 'purple')



plt.show()
# With a house which Living Area is 4000, print predict Price

Living_Area = [[4000]]

Price = model.predict(Living_Area)



print(Price)