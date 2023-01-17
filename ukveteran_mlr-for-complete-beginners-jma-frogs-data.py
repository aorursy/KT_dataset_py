import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
df=pd.read_csv("../input/frogs-data/frogs.csv")

df.head()
df.drop(["Unnamed: 0"],axis=1,inplace=True)

df.head()
x=df.iloc[:,[0,1]].values #northing, easting

y=df.altitude.values.reshape(-1,1)
multiple_linear_regression=LinearRegression()

multiple_linear_regression.fit(x,y)
print("b0: ",multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)
df.describe().T
multiple_linear_regression.predict(np.array([[8,5],[8,8]]))