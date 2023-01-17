# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
df
plt.scatter(df.Level,df.Salary)
X=df.iloc[:,1:2].values

y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
Lr = LinearRegression()

Lr.fit(X_train,y_train)
y_pred=Lr.predict(X_test)
print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
plt.plot(X_test,Lr.predict(X_test),label="Best fit curve",color='r')

plt.scatter(X,y)

plt.legend()

plt.show()
def polynomialRegression(X,y,k=14):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=False)



    poly = PolynomialFeatures(degree=k)

    X_poly = poly.fit_transform(X_train)

    lr = LinearRegression()

    lr.fit(X_poly,y_train)

  

    X_test_poly =poly.fit_transform(X_test)

    y_pred=lr.predict(X_test_poly)



    training_score = r2_score(y_train, lr.predict(X_poly))

    test_score = r2_score(y_test,y_pred)

    

    print("r2 score train : " ,training_score)

    print("r2 score test : " ,test_score)

  

    return training_score, test_score
accuracy = {}

for i in range(1,20):

    print('for degree = ',i)

    train_score,test_score=polynomialRegression(X,y,i)

    accuracy[i]=test_score

    print('\n')

max_accu = max(accuracy.values())

for k,v in accuracy.items():

    if v==max_accu:

        print(f'degree {k} gives the highest test accuracy')