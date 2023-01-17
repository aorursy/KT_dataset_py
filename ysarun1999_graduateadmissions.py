import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(12,8))

sns.distplot(df['Chance of Admit '],bins=40)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),cmap='magma',annot=True)
plt.figure(figsize=(12,8))

sns.regplot(x='GRE Score',y='Chance of Admit ',data=df)
plt.figure(figsize=(12,8))

sns.regplot(x='TOEFL Score',y='Chance of Admit ',data=df)
plt.figure(figsize=(12,8))

sns.regplot(x='CGPA',y='Chance of Admit ',data=df)
X = df.drop(['Chance of Admit ','Research','LOR ','Serial No.'],axis=1)

y = df['Chance of Admit ']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
pred = lm.predict(X_test)
plt.figure(figsize=(12,8))

plt.scatter(y_test,pred)
from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_test,pred)))
from sklearn.metrics import r2_score

print(r2_score(y_test,pred))
from sklearn.ensemble import RandomForestRegressor

rm = RandomForestRegressor()

rm.fit(X_train,y_train)
pred = rm.predict(X_test)

plt.figure(figsize=(12,8))

plt.scatter(y_test,pred)
print(np.sqrt(mean_squared_error(y_test,pred)))
print(r2_score(y_test,pred))