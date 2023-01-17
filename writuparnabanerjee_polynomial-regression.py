# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/position-salaries/Position_Salaries.csv')
df.head()
df.shape
plt.scatter(df.Level, df.Salary)
X=df.iloc[:,1:2].values

y=df.iloc[:,2:3].values
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

y=scaler.fit_transform(y)
y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.30, random_state=0)
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=3)
X_poly=poly.fit_transform(X_train)

print(X_poly.shape)

X_poly
from sklearn.linear_model import LinearRegression

l=LinearRegression()
l.fit(X_poly,y_train)
X_test_poly =poly.fit_transform(X_test)
y_pred=l.predict(X_test_poly)
from sklearn.metrics import r2_score,mean_squared_error

print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
plt.scatter(X_test,y_pred, label='Model', color='red')

plt.scatter(X_train,y_train, label='Data', marker='x', color='green')

plt.legend()

plt.show()
def polynomial_regression(X,y,degree):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    poly=PolynomialFeatures(degree=degree)

    X_poly=poly.fit_transform(X_train)

    l=LinearRegression()

    l.fit(X_poly,y_train)

    

    X_test_poly =poly.fit_transform(X_test)

    y_pred=l.predict(X_test_poly)

    training_score=r2_score(y_train,l.predict(X_poly))

    test_score=r2_score(y_test,y_pred)

    return training_score, test_score

    
train=[]

test=[]

for i in range(0,10):

    r2train,r2test=polynomial_regression(X,y,degree=i)

    train.append(r2train)

    test.append(r2test)

x=np.arange(10)

plt.plot(x,train,label="Training")

plt.plot(x,test,label="Test")

plt.legend()

plt.xlabel("Degree")

plt.ylabel("r2-Score")

plt.title("R2-Score");

plt.show()
#Therefore, the best value for degree parameter is 5.