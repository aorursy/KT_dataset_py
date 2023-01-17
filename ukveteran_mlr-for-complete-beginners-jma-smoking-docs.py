import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df= pd.read_csv('../input/smoking-deaths-among-doctors/breslow.csv')

df
df.drop(["Unnamed: 0"],axis=1,inplace=True)

df.head()
x=df.iloc[:,[0,1]].values #n, y

y=df.ns.values.reshape(-1,1)
multiple_linear_regression=LinearRegression()

multiple_linear_regression.fit(x,y)
print("b0: ",multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)
df.describe().T
multiple_linear_regression.predict(np.array([[8,5],[8,8]]))