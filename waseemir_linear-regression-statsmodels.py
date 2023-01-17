import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

sns.set()
data=pd.read_csv('../input/advertising-dataset/advertising.csv')

data.describe()
#dependent variable=Sales, independent variable=TV ad

s,t1,r,n=data['Sales'],data['TV'],data['Radio'],data['Newspaper']

plt.scatter(t1,s)

plt.xlabel('TV')

plt.ylabel('Sales')

plt.show()
plt.scatter(r,s)

plt.xlabel('Radio')

plt.ylabel('Sales')

plt.show()
plt.scatter(n,s)

plt.xlabel('Newspaper')

plt.ylabel('Sales')

plt.show()
t=sm.add_constant(t1)

results=sm.OLS(s,t).fit()

results.summary()
plt.scatter(t1,s)

yhat=6.9748+0.0555*t1

plt.plot(t1,yhat,c='orange',label='Regression')

plt.xlabel('TV')

plt.ylabel('Sales')

plt.show()