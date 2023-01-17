# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/linearregression/1.1 linear_regression_dataset.csv.csv",sep = ";")
df.columns
df.info()
df.head()

# The first 5 samples of the data set
df.head(10)
df.tail()

# The last 5 samples of the data set
df.dtypes
# plot data

plt.scatter(df.deneyim,df.maas)

plt.xlabel("deneyim")

plt.ylabel("maas")

plt.show()
# sklearn library

from sklearn.linear_model import LinearRegression



# linear regression model

linear_reg = LinearRegression()



x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)



linear_reg.fit(x,y)
#%% prediction

import numpy as np



#b0 = linear_reg.predict(0)

#print("b0: ",b0)



b0 = linear_reg.intercept_

print("b0: ",b0)   # y eksenini kestigi nokta intercept



b1 = linear_reg.coef_

print("b1: ",b1)   # egim slope
# maas = 1663 + 1138*deneyim 

maas_yeni = 1663 + 1138*11

print(maas_yeni)
print(linear_reg.predict([[11]]))
# visualize line

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim





plt.scatter(x,y)





y_head = linear_reg.predict(array)  # maas



plt.plot(array, y_head,color = "red")



plt.show()

linear_reg.predict([[100]])

df=pd.read_csv("../input/multiplelinear/multiple_linear_regression_dataset.csv.csv",sep = ";")
df.head()
x = df.iloc[:,[0,2]].values

y = df.maas.values.reshape(-1,1)
# %% fitting data

from sklearn.linear_model import LinearRegression

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)
print("b0: ", multiple_linear_regression.intercept_)

print("b1,b2: ",multiple_linear_regression.coef_)
# predict

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))
array=np.array([[1,23],[5,29],[10,34],[12,38],[4,25],[15,40]])

array
import matplotlib.pyplot as plt

y_head=multiple_linear_regression.predict(array)



plt.scatter(x[:,0],y)

plt.scatter(x[:,1],y,color="green")



plt.plot(array,y_head,color="red")

plt.show()



multiple_linear_regression.predict(np.array([[9,35]]))
df=pd.read_csv("../input/polynomialregression/polynomial regression.csv",sep = ";")
df.head()
df.info()
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
y = df.araba_max_hiz.values.reshape(-1,1)

x = df.araba_fiyat.values.reshape(-1,1)
plt.scatter(x,y)

plt.ylabel("araba_max_hiz")

plt.xlabel("araba_fiyat")

plt.show()
lr = LinearRegression()



lr.fit(x,y)
y_head = lr.predict(x)
plt.plot(x,y_head,color="red",label ="linear")

plt.show()

print("10 milyon tl lik araba hizi tahmini: ",lr.predict([[10000]]))
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)



x_polynomial = polynomial_regression.fit_transform(x)
x

x_polynomial
# %% fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)
y_head2 = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head2,color= "green",label = "poly")

plt.legend()

plt.show()
print(linear_regression2.predict(polynomial_regression.fit_transform([[1600]])))
df = pd.read_csv("../input/decisiontreeregression/decisiontreeregressiondataset.csv",sep = ";")
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
#%%  decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()   # random sate = 0

tree_reg.fit(x,y)
#tree_reg.predict(5)
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict([[ x_]])
# %% visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color = "green")

plt.xlabel("tribun level")

plt.ylabel("ucret")

plt.show()
df = pd.read_csv("../input/decisiontreeregression/decisiontreeregressiondataset.csv",sep = ";",header = None)
df.head()
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x,y)
print("7.8 seviyesinde fiyatın ne kadar olduğu: ",rf.predict([[7.8]]))