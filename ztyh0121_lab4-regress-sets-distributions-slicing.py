import sklearn.datasets as skldat

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

X,Y=skldat.make_regression(n_samples=1000,n_features=100,n_informative=10,n_targets=1,noise=1,random_state=2118)

X.shape
Y.shape
type(X)
type(Y)
linmod=linear_model.LinearRegression()

linmod.fit(X,Y)
linmod.coef_
linmod.intercept_
Ypred=linmod.predict(X)
Ypred
max(linmod.coef_)
np.argmax(linmod.coef_)
np.where(linmod.coef_>0.1)
mean_squared_error(Y,Ypred)
Y
r2_score(Y,Ypred)
linmod.fit(X[:,65].reshape(-1,1),Y)

Ypred=linmod.predict(X[:,65].reshape(-1,1))

linmod.coef_
X[:,65].shape
X[:,65].reshape(-1,1).shape
plt.scatter(X[:,65],Y, alpha=0.3, s=10)

plt.plot(X[:,65],Ypred, c="r")

plt.show()
import statsmodels.formula.api as sm

linmod = sm.OLS( Y, X ).fit()

linmod.summary()
linmod.params
linmod.mse_resid
linmod.nobs
linmod.df_model
S={1,2,3,4,5,6}

A = {x for x in S if x%2==1}

B = {x for x in S if x>3}

S
A
B
A&B
A|B
A-B
A^B
1 in A
2 not in B
set([1,2,3,4,5,6])
exlist=[[1,2,3,4,5],[6,7,8,9,0],[9,7,5,3,1]]

exnparray=np.random.rand(5,3)

expdarray=pd.DataFrame(exnparray)

exlist
exnparray
expdarray
type(exlist)
type(exnparray)
type(expdarray)
exlist[0]
exlist[0][2:]
exlist[0][2::-1]
exlist[0][::2]
exnparray[::-1,::-1]
exnparray[0::2,0::2]
exnparray[2:4:,1:3:]
exnparray[2:4,1:3]
expdarray.iloc[0:2,0:2]