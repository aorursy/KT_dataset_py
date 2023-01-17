import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
data = pd.read_csv("../input/Position_Salaries.csv")



data.head()
x=data.iloc[:,1:2]

x
y=data.iloc[:,2]

y
model = DecisionTreeRegressor(random_state=0)

model.fit(x,y)
a = model.predict(x)
mae = mean_absolute_error(y,a)



mae
fig,ax = plt.subplots()



ax.scatter(x,y)

ax.plot(x,a)



plt.xlabel("Position Level")

plt.ylabel("Salary")



plt.show()