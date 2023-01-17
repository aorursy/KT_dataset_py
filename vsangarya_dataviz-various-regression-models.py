

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

%matplotlib inline


df=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")
df.head()
df.shape
df.isna().any()

fdf = df[['year', 'km_driven','fuel', 'seller_type', 'transmission', 'owner','selling_price']]
fdf.head()
plt.figure(1, figsize=(8, 6))
plt.bar(fdf.year,fdf.selling_price, color='blue',alpha=0.4)
plt.xlabel("Year")
plt.ylabel("Selling price")
plt.show()
plt.figure(1, figsize=(10, 12))
plt.scatter(fdf.km_driven,fdf.selling_price, color='red',alpha=0.2)
plt.xlabel("Kms driven")
plt.ylabel("Selling price")
plt.show()
plt.figure(1, figsize=(10, 10))
plt.scatter(fdf.seller_type,fdf.selling_price, color='orange',alpha=0.2)
plt.xlabel("Seller Type")
plt.ylabel("Selling price")
plt.show()
plt.figure(1, figsize=(10, 8))
plt.scatter(fdf.owner,fdf.selling_price, color='green',alpha=0.2)
plt.xlabel("Owner")
plt.ylabel("Selling price")
plt.show()
print(fdf['fuel'].unique())
print(fdf['seller_type'].unique())
print(fdf['transmission'].unique())
print(fdf['owner'].unique())
X=fdf.values
x=X[:,0:6]
y=X[:,6]
x[0:3]
f=preprocessing.LabelEncoder()
f.fit(['Petrol','Diesel','CNG','LPG','Electric'])
x[:,2]=f.transform(x[:,2])


s=preprocessing.LabelEncoder()
s.fit(['Individual','Dealer','Trustmark Dealer'])
x[:,3]=s.transform(x[:,3])

t=preprocessing.LabelEncoder()
t.fit(['Manual','Automatic'])
x[:,4]=t.transform(x[:,4])

o=preprocessing.LabelEncoder()
o.fit(['Test Drive Car','First Owner' ,'Second Owner', 'Third Owner','Fourth & Above Owner' ])
x[:,5]=o.transform(x[:,5])
x[0:3]
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x[0:2]
xtrain1,xtest1,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=3)
print(xtrain1.shape,ytrain.shape,xtest1.shape,ytest.shape)
poly=PolynomialFeatures(degree=2)
xtrain=poly.fit_transform(xtrain1)
xtest=poly.fit_transform(xtest1)
Lr2=linear_model.LinearRegression()
Lr2.fit(xtrain1,ytrain)

Lr=linear_model.LinearRegression()
Lr.fit(xtrain,ytrain)
yhat1=Lr.predict(xtest)
print("R2 score :",r2_score(ytest,yhat1))
sns.distplot(ytest-yhat1, color='red')
Br=linear_model.BayesianRidge()
Br.fit(xtrain,ytrain)
yhat2=Br.predict(xtest)
print("R2 score :",r2_score(ytest,yhat2))
sns.distplot(ytest-yhat2, color='blue')

L=linear_model.Lasso(alpha=0.2)
L.fit(xtrain,ytrain)
yhat3=L.predict(xtest)
print("R2 score :",r2_score(ytest,yhat3))
sns.distplot(ytest-yhat3, color='green')
R=linear_model.Ridge(alpha=.5)
R.fit(xtrain,ytrain)
yhat4=R.predict(xtest)
print("R2 score :",r2_score(ytest,yhat4))
sns.distplot(ytest-yhat4, color='yellow')

importance = abs(Lr2.coef_)
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' %(i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()