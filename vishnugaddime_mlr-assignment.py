import pandas as pd

import numpy as np

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

import math



df = pd.read_csv('../input/amsprediction/amsPredictionSheet11-201010-101537.csv')

df.head()
df.describe()
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
endog = df['ESE']

exog = sm.add_constant(df[['MSE','Attendance','HRS']])

print(exog)
X=exog.to_numpy()

Y=endog.to_numpy()

s1_xt= np.transpose(X)

print(s1_xt)
s2_null= np.matmul(s1_xt,X)

print(s2_null)
s3_inv=np.linalg.inv(s2_null)

print(s3_inv)
s4_mul= np.matmul(s3_inv, s1_xt)

print(s4_mul)
s5_res= np.matmul(s4_mul,Y)

print(s5_res)
mod= sm.OLS(endog, exog)

results = mod.fit()

print (results.summary())
from sklearn import linear_model

X = df[['MSE','Attendance','HRS']]

Y = df['ESE']



lm = linear_model.LinearRegression()

model = lm.fit(X,Y)

lm.coef_
lm.intercept_