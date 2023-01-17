import numpy as np 

import pandas as pd 

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

import math
df = pd.read_csv('../input/iris/Iris.csv')

df.drop('Id', inplace=True, axis=1)

df.drop('Species', inplace=True, axis=1)

df.head()
df.describe()
correlation=df.corr()

correlation.style.background_gradient(cmap='coolwarm')
dep = df['PetalWidthCm']

indep = sm.add_constant(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm']])

print(indep)
mod = sm.OLS(dep,indep)

results = mod.fit()

print (results.summary())
#coverting to arrays to perform matrix operations

X = indep.to_numpy()

Y = dep.to_numpy()

#calculating x transpose

x_transpose =np.transpose(X) 

print(x_transpose)
#Multiplying x_transpose with x

x_transposex= np.matmul(x_transpose,X)

print(x_transposex)
#Calculating inverse of x_transpose*x

inv = np.linalg.inv(x_transposex)

print(inv)
#Again Multiplying inverse with x_transpose

H= np.matmul(inv,x_transpose)

print(H)
#Multiplying H with Y

result=np.matmul(H,Y)

print(result)
def predict_y(x1,x2,x3):

    y_hat = result[0] + result[1]*x1 + result[2]*x2 + result[3]*x3

    return y_hat;
PetalWidth_pre = predict_y(df['SepalLengthCm'], df['SepalWidthCm'], df['PetalLengthCm'])

PetalWidth_pre
r_square = 1 - sum((dep-PetalWidth_pre)*(dep-PetalWidth_pre))/sum((dep-dep.mean())*(dep-dep.mean()))

r_square
def RSE(y_true, y_pred):

   

    y_true = np.array(y_true)

    y_predicted = np.array(y_pred)

    RSS = np.sum(np.square(y_true - y_pred))



    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse





rse= RSE(df['PetalWidthCm'],PetalWidth_pre)

print(rse)
from sklearn import linear_model

linmod = linear_model.LinearRegression()

model = linmod.fit(indep,dep)

print(linmod.coef_)

print(linmod.intercept_)
plt.scatter(df['SepalLengthCm'], dep)

plt.scatter(df['SepalLengthCm'], PetalWidth_pre)

plt.show()
plt.scatter(df['SepalWidthCm'], dep)

plt.scatter(df['SepalWidthCm'], PetalWidth_pre)

plt.show()
plt.scatter(df['PetalLengthCm'], dep)

plt.scatter(df['PetalLengthCm'], PetalWidth_pre)

plt.show()