# Import all libraries we need

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization

# sklearn library

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

# Evaluation Metric

from sklearn.metrics import r2_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

df.head()
df.info()
# plot data

plt.scatter(df["Social support"],df["Score"])

plt.xlabel("Social Support")

plt.ylabel("Score")

plt.show()

# linear regression model

linear_reg = LinearRegression()



x = df["Social support"].values.reshape(-1,1)

y = df["Score"].values.reshape(-1,1)



linear_reg.fit(x,y)
Predicted_Score1 = linear_reg.predict([[1.5]])

print("Predicted Score 1: ",Predicted_Score1)



Predicted_Score2 = linear_reg.predict([[0.5]])

print("Predicted Score 2: ",Predicted_Score2)



Predicted_Score3 = linear_reg.predict([[1.8]])

print("Predicted Score 3: ",Predicted_Score3)



intercept = linear_reg.intercept_

print("intercept: ",intercept)   # y eksenini kestigi nokta intercept



slope = linear_reg.coef_

print("slope: ",slope)   # egim slope



# Score = 1.91243024 + 2.89098704*Social Support 



y_predicted = linear_reg.predict(x)

plt.scatter(x,y)

plt.plot(x, y_predicted,color = "red")

plt.xlabel("Social Support")

plt.ylabel("Score")

plt.title("Linear Regression")

plt.show()

print("r_score: ", r2_score(y,y_predicted))
# Multiple Linear Regression Model

x = df.iloc[:,3:].values

y = df["Score"].values.reshape(-1,1)

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)
print("Intercept: ", multiple_linear_regression.intercept_)

print("b1,b2,b3,b4,b5,b6: ",multiple_linear_regression.coef_)
# prediction

multiple_linear_regression.predict(np.array([[1.340,1.587,0.986,0.596,0.153,0.393]]))
df.head()
# plot data

plt.scatter(df["Social support"],df["Score"])

plt.xlabel("Social support")

plt.ylabel("Score")

plt.show()
x = df["Healthy life expectancy"].values.reshape(-1,1)

y = df["Score"].values.reshape(-1,1)
lr = LinearRegression()

lr.fit(x,y)

y_head = lr.predict(x)

plt.scatter(df["Healthy life expectancy"],df["Score"])

plt.xlabel("Healthy life expectancy")

plt.ylabel("Score")

plt.plot(x,y_head,color="red",label ="linear")

plt.show()
polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)
y_head2 = linear_regression2.predict(x_polynomial)

plt.scatter(df["Healthy life expectancy"],df["Score"])

plt.xlabel("Healthy life expectancy")

plt.ylabel("Score")

plt.plot(x,y_head2,color= "green",label = "poly")

plt.title("Polynomial Regression")

plt.legend()

plt.show()

print("r_square score: ", r2_score(y,y_head2))
x = df["GDP per capita"].values.reshape(-1,1)

y = df["Score"].values.reshape(-1,1)
tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

tree_reg.predict([[1.2]])

x_ = np.arange(min(x),max(x),0.1).reshape(-1,1)

y_head = tree_reg.predict(x_)
# visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color = "green")

plt.xlabel("GDP per capita")

plt.ylabel("Score")

plt.title("Decision Tree")

plt.show()
x = df["Freedom to make life choices"].values.reshape(-1,1)

y = df["Score"].values.reshape(-1,1)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x,y)

print("Predicted Value = : ",rf.predict([[0.5]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)
# visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="green")

plt.xlabel("Freedom to make life choices")

plt.ylabel("Score")

plt.show()