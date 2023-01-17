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
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/position-salaries/Position_Salaries.csv')
data
data.shape
#graph to see the variation of the given data



plt.scatter(data.Level, data.Salary)
X = data.iloc[:,1:2]

y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#Linear Regression



from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
#R2 Score calculation



from sklearn.metrics import r2_score



print("R2 score", r2_score(y_test,y_pred))
#graph showing the best fit line for Linear Regression



plt.plot(X_test,LR.predict(X_test),label="Modal")

plt.scatter(X_train,y_train,label="data",color="r")

plt.legend()

plt.show()
#Function to execute polynomial regression



from sklearn.preprocessing import PolynomialFeatures

def polynomialRegression(X,y,k=14):



  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



  poly = PolynomialFeatures(degree=k)

  X_poly = poly.fit_transform(X_train)

  lr = LinearRegression()

  lr.fit(X_poly,y_train)

  

  X_test_poly =poly.fit_transform(X_test)

  y_pred=lr.predict(X_test_poly)



  training_score = r2_score(y_train, lr.predict(X_poly))

  test_score = r2_score(y_test,y_pred)

  

  return training_score, test_score
#graph to show the variations of r2 score for different values of polynomial degree



train = []

test = []

for i in range(1,10):

  r2train,r2test=polynomialRegression(X,y,k=i)

  train.append(r2train)

  test.append(r2test)

x=np.arange(9)+1

plt.plot(x,train,label="Training")

plt.plot(x,test,label="Test")

plt.legend()

plt.xlabel("Degree")

plt.ylabel("r2-Score")

plt.title("R2-Score");

plt.show()
#Calculating the r2 score from the graph



test[4]