import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm 

import math

df = pd.read_csv("../input/predicting-ese-marks/amsPredictionSheet1-201009-150447.csv")

df.head()
df.describe()
corr=df.corr()

corr.style.background_gradient(cmap='coolwarm')
endog = df['ESE']

exog = sm.add_constant(df[['MSE','Attendance','HRS']])

print(exog)
x = exog.to_numpy()

y = endog.to_numpy()

s1_xt=np.transpose(x)

print(s1_xt)
s2_mull=np.matmul(s1_xt,x)

print(s2_mull)
s3_inv = np.linalg.inv(s2_mull)

print(s3_inv)
s4_mul = np.matmul(s3_inv,s1_xt)

print(s4_mul)
s5_res = np.matmul(s4_mul,y)

print(s5_res)
mod = sm.OLS(endog, exog)

results = mod.fit()

print (results.summary())
def RSE(y_true, y_predicted):

   

    y_true = np.array(y_true)

    y_predicted = np.array(y_predicted)

    RSS = np.sum(np.square(y_true - y_predicted))



    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse
yp= results.predict()

ypa = np.array(yp)

yta = df['ESE']

eterms =yta-ypa





df1 = pd.DataFrame(eterms)

df1['ESE'].hist(bins=10)
rse= RSE(df['ESE'],results.predict())

print(rse)

from sklearn import linear_model

X = df[['MSE','Attendance','HRS']]

y = df['ESE']



lm = linear_model.LinearRegression()

model = lm.fit(X,y)

lm.coef_
lm.intercept_