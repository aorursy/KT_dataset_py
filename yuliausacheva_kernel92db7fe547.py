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
data = pd.read_csv("/kaggle/input/coursera-ml/ex1data1.txt",

header=None, names=["population", "profit"])
data2.head()
from.sklearn.preprocessing import StandardScaler
x = data2.drop("price", axis=1)

y = data2["price"]
from sklearn.preprocessing import StandardScaler

x = data2.drop("price", axis=1)

y = data2["price"]

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
lr = LinearRegression()

lr.fit(x_scaled,y)
from sklearn.preprocessing import StandardScaler

x = data["population"]

y = data["profit"]

import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(x,y,"rx");

k = 0.8

b = 0

h = k*x+b

plt.plot(x,h);
data2 = pd.read_csv("/kaggle/input/coursera-ml/ex1data2.txt",

header=None, names=["square", "rooms", "price"])
from sklearn.linear_model import LinearRegression
lr.fit(np.array(x).reshape(-1, 1),y)
lr.score(np.array(x).reshape(-1, 1),y)
print(lr.intercept_)

print(lr.coef_)

h = lr.coef_*x + lr.intercept_
plt.plot(x,y,"rx");

plt.plot(x,h);