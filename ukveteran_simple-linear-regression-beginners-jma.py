import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/cost-function-for-electricity-producers/Electricity.csv")

df.head()
df.corr()
df.describe().T
linear_reg=LinearRegression()

x=df.cost.values.reshape(-1,1)

y=df.q.values.reshape(-1,1)



linear_reg.fit(x,y)



b0=linear_reg.predict([[4]])

print("========================")

print("b0: ",b0)



b1=linear_reg.coef_

print("========================")

print("b1: ",b1)



array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]).reshape(-1,1)



y_head=linear_reg.predict(array)

plt.scatter(x,y)

plt.plot(array,y_head,color='red')

plt.xlabel("Cost")

plt.ylabel("q")

plt.show()
linear_reg.predict([[9]])