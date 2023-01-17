import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import mean_squared_error

df=pd.read_csv('../input/daksh2k19-mgyan/train.csv')

df1=df['dteday'].str.split(pat='/',expand=True)

df1.columns=['month','day','year']

df=df.join(df1)

df.drop(['year','mnth','dteday','casual'],axis=1,inplace=True)

df['day']=pd.to_numeric(df['day'])

df['month']=pd.to_numeric(df['month'])

df.head()
mf=pd.read_csv('../input/daksh2k19-mgyan/test.csv')

mf1=mf['dteday'].str.split(pat='/',expand=True)

mf1.columns=['month','day','year']

mf=mf.join(mf1)

mf.drop(['year','mnth','dteday','casual'],axis=1,inplace=True)

mf['day']=pd.to_numeric(mf['day'])

mf.head()
plt.figure(figsize=(18,5))

sns.heatmap(df.corr(),annot=True)
sns.boxplot(df['registered'])
plt.figure(figsize=(18,5))

sns.heatmap(df.corr(),annot=True)
x=df[['instant','season','holiday','weekday','workingday','temp','atemp','hum','windspeed','registered','yr','month','hr','weathersit']]
xtest=mf[['instant','season','holiday','weekday','workingday','temp','atemp','hum','windspeed','registered','yr','month','hr','weathersit']]
y=df['cnt']
from sklearn.model_selection import train_test_split

x1,x2,y1,y2=train_test_split(x,y,test_size=0.001,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x1,y1)
yh=lr.predict(x2)
yh2=lr.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(y2,yh)
from sklearn.metrics import r2_score

a2=r2_score(y2,yh)

a2
import seaborn as sns

sns.residplot(yh,y2)
sns.regplot(yh,y2)






#poly
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=4)
x_poly=pr.fit_transform(x1)
x_polytest=pr.fit_transform(x2)
x_polytest1=pr.fit_transform(xtest)
lr.fit(x_poly,y1)
yhp=lr.predict(x_polytest)
yhptest=lr.predict(x_polytest1)
mean_squared_error(y2,yhp)
yh3=yhptest*yhptest

yh4=np.sqrt(yh3)

yh4[0:5]
mf['cnt']=yh4.astype(int)

mf.head()
ff=mf[['instant','cnt']]

ff.set_index('instant',inplace=True)

ff.head()
ff.to_csv('kaggle1')