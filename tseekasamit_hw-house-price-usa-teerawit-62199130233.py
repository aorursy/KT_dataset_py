import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/housedata/data.csv')
df.head()

df.info()

df.shape
#move col 'price' to last col

df1 = df[[x for x in df if x not in ['price']] + ['price']]
list(df1)
df1.describe(include='number')
df1.describe(include='object')
#check NULL

print(df1.isnull().sum().sum())

print(df1.isnull().any())

import missingno as msno

msno.matrix(df1)
def changeTypes(x):

    df1[x] = df1[x].astype(float,errors='ignore')

    

changeTypes('sqft_living')

changeTypes('sqft_lot')

changeTypes('waterfront')

changeTypes('view')

changeTypes('condition')

changeTypes('sqft_above')

changeTypes('sqft_basement')

changeTypes('yr_built')

changeTypes('yr_renovated')
df1.dtypes
df1['floors'].value_counts()
#https://seaborn.pydata.org/generated/seaborn.jointplot.html

#http://ging-ks.blogspot.com/2019/02/python-plot.html

sns.jointplot('price', 'sqft_living', data=df1, height=5, ratio=3, color='g', kind='reg')
sns.jointplot('floors', 'bathrooms', data=df, color='b', kind='hex')

sns.jointplot('floors', 'bedrooms', data=df, color='r', kind='hex')
#https://seaborn.pydata.org/generated/seaborn.pairplot.html?highlight=pairplot#seaborn.pairplot

#http://ging-ks.blogspot.com/2019/02/python-plot.html

sns.pairplot(df1[['price','sqft_living','bedrooms','bathrooms','floors']], diag_kind="kde")
df1.shape
df1.drop(['date', 'street', 'statezip', 'country', 'waterfront', 'view','sqft_basement', 'yr_renovated','city'], axis=1, inplace=True)
X = df1.iloc[:, 0:-1]

y = df1.iloc[:, -1]

print(X.shape, y.shape)
df1
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer(

    [('One_hot_encoder', OneHotEncoder(), [5])],  #[0 ,1] multi column  #The column neumber 

    remainder = 'passthrough'                      #Leave the rest of t

)



X = np.array(ct.fit_transform(X), dtype = np.float)

X = X[:, 1:] 
X
y
#c_columns = ['condition']

# Perform one-hot encoding

#for column in c_columns:

#  dummies = pd.get_dummies(X[column], prefix=column, drop_first=False)

#  X = pd.concat([X, dummies], axis=1)



#X.shape
#Drop original categorical columns

#X.drop(['condition'], inplace=True, axis=1)

#X.shape
#X = X.values

#y = y.values
#X = X[:, 1:]
X.shape
#make train/test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
#Feature Scaling

#from sklearn.preprocessing import StandardScaler

#sc_X= StandardScaler()

#X_train= sc_X.fit_transform(X_train)

#X_test= sc_X.transform(X_test)
#Linear Regression

#Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#Predicting the Test Set Results

y_pred = regressor.predict(X_test)
import statsmodels.api as sm

X = np.append(arr=np.ones((4600,1)).astype(int), values = X, axis=1)#Add 1 to the 1st column
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11]] #Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[1,2,3,4,5,7,8,9,10,11]] #Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[1,2,3,4,5,7,8,9,11]] #Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[2,3,4,5,7,8]] #Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()