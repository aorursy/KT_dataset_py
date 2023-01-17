# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df = pd.read_csv('../input/the-housing-data/Housing_data.csv',sep=';')
df.head()
X = df.drop(['price'] , axis=1)
Y = df[['price']]
print(Y['price'].min())
print(Y['price'].max())
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)
from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()
lmodel.fit(xtrain,ytrain)
Yp_train = lmodel.predict(xtrain)
Yp_test = lmodel.predict(xtest)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(ytrain,Yp_train))
print(mean_absolute_error(ytest,Yp_test))
X = df.drop(['price'] , axis=1)
Y = df[['price']]
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree = 3)

pol.fit(X)
Xp = pol.transform(X)
from sklearn.model_selection import train_test_split
xtrain_p , xtest_p , ytrain_p , ytest_p = train_test_split(Xp,Y)
print(X.shape)
print(Xp.shape)
lmodel_p = LinearRegression()
lmodel_p.fit(xtrain_p,ytrain_p)
Ypp_train = lmodel_p.predict(xtrain_p)
Ypp_test = lmodel_p.predict(xtest_p)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(Ypp_train,ytrain_p))
print(mean_absolute_error(Ypp_test,ytest_p))
from sklearn.model_selection import train_test_split
xtrain_p , xtest_p , ytrain_p , ytest_p = train_test_split(X,Y)

training_error = []
testing_error = []
for i in range(1,10):
    pol = PolynomialFeatures(degree = i)

    xtrain_p = pol.fit_transform(xtrain_p)
    xtest_p = pol.fit_transform(xtest_p)
    
    lmodel_p = LinearRegression()
    lmodel_p.fit(xtrain_p,ytrain_p)
    
    Ypp_train = lmodel_p.predict(xtrain_p)
    Ypp_test = lmodel_p.predict(xtest_p)
    
    training_error.append(mean_absolute_error(Ypp_train,ytrain_p))
    testing_error.append(mean_absolute_error(Ypp_test,ytest_p))
