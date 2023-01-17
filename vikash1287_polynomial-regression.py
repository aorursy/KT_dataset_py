import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/Position_Salaries.csv")
dataset.head(2)
x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

plt.scatter(x,y,color='red')
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(x)
x_poly
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_poly,y)
plt.scatter(x,y,color = 'red')

plt.plot(x,regressor.predict(x_poly),color ='blue')

plt.xlabel("Experiance ")

plt.ylabel("Salary data ")

plt.title("Degree 4")