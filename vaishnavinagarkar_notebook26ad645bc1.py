import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from pandas import DataFrame

import statsmodels.api as sm
df=pd.read_csv('../input/lab-mlr/amsPredictionSheet11-201010-101537.csv')

df.head() 
x=df[['Attendance', 'MSE']]

y=df['ESE']

regr=linear_model.LinearRegression()

regr.fit(x,y)
print(regr.coef_)
Y_predicted=regr.predict(x)

Y_predicted 
plt.scatter(df['Attendance'], df['MSE'],color='black')

plt.title('Attendance Vs MSE',fontsize=15)

plt.xlabel('Attendance', fontsize=15)

plt.ylabel('MSE',fontsize=15)

plt.grid(True)

plt.show() 
df.head() 
x=df[['Attendance','MSE']]

y=df['ESE']

x1=sm.add_constant(x)

est=sm.OLS(y,x1).fit()

est.summary()
df_sub=df[['Attendance','MSE']]

df_sub.corr() 
corrmat=df_sub.corr()

f, ax=plt.subplots(figsize=(6,6))

sns.heatmap(corrmat,vmax=1,square=True);

plt.show() 
fig_size=plt.rcParams["figure.figsize"]

fig_size[0]=15

fig_size[1]=5

df_sub.hist(bins=60)

plt.show() 
my_data=pd.read_csv('../input/lab-mlr/amsPredictionSheet11-201010-101537.csv')

print(df.columns)

print(df.head(7))                   
pd.isna(df).any()
#mean_normalization

my_data=(my_data-my_data.mean())/my_data.std()

my_data.head()
x=my_data.iloc[:,0:2]

ones=np.ones([x.shape[0],1])

x=np.concatenate((ones,x),axis=1)    



y=my_data.iloc[:,2:3].values

theta=np.zeros([1,3])



alpha=0.01

iters=1000

def computeCost(x,y,theta):

    tobesummed=np.power(((x @ theta.T)-y),2)

    return np.sum(tobesummed)/(2*len(x))
def gradientDescent(x,y,theta,iters,alpha):

    cost=np.zeros(iters)

    for i in range(iters):

        theta=theta-(alpha/len(x))*np.sum(x*(x @ theta.T-y),axis=0)

        cost[i]=computeCost(x,y,theta)

        return theta,cost
g,cost=gradientDescent(x,y,theta,iters,alpha)

print(g) 

finalCost=computeCost(x,y,g)

print(finalCost) 
#plotting cost

fig, ax=plt.subplots()

ax.plot(np.arange(iters),cost,'b')

ax.set_xlabel('Attendance')

ax.set_ylabel('HRS')

ax.set_title('Attendance Vs HRS') 