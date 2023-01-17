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
data=pd.read_csv('../input/years-of-experience-and-salary-dataset/Salary_Data.csv')
data
data.shape
#y=mx+c have to find m and c
X=np.array(data.iloc[:,0].values)
y=np.array(data.iloc[:,1].values)
X_mean=np.mean(X)
y_mean=np.mean(y)
m_numerator=(X-X_mean)*(y-y_mean)
m_denominator=(X-X_mean)**2
m=m_numerator.sum()/m_denominator.sum()
c=y_mean-(m*X_mean)
round(m,2)
round(c,2)
class LinearRegression:
    def __init__(self):
        self.numerator=0
        self.denominator=0
        self.m=0
        self.c=0
    def fit(self,X,y):
        X_mean=np.mean(X)
        y_mean=np.mean(y)
        N=len(X)
        for i in range(N):
            self.numerator+=(X[i]-X_mean)*(y[i]-y_mean)
            self.denominator+=(X[i]-X_mean)**2
        self.m+=self.numerator/self.denominator
        self.c+=y_mean-(self.m*X_mean)
        return self
    def predict(self,X):
        return self.m*X+self.c
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
y_train.shape
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)*100
print("The accuracy of the model is : ",accuracy)
