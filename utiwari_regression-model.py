import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from random import seed

from random import gauss
X=[]

Y=[]

for i in range(30):

    X.append(i+gauss(0,2))

    Y.append(i+gauss(0,2))

plt.scatter(X,Y)
x = np.array(X).reshape(1,-1)

y = np.array(Y).reshape(1,-1)

model =LinearRegression()

model.fit(x,y)
X=[i for i in range(30)]

Y=[i for i in range(30)]

x = np.array(X).reshape(1,-1)

y = np.array(Y).reshape(1,-1)

plt.plot(X,Y)

z=model.predict(x)

plt.scatter(z,y)