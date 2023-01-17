# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/years-of-experience-and-salary-dataset/Salary_Data.csv')
df.head()
df.info()
X=df.iloc[:,0].values

Y=df.iloc[:,1].values

plt.scatter(X,Y)

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()

Xbar=np.mean(X)

Ybar=np.mean(Y)

var1=(X-Xbar)*(Y-Ybar)

var2=(X-Xbar)**2

m=(sum(var1)/sum(var2))

b=Ybar-(m*Xbar)

y1=m*X+b

plt.scatter(X,Y)

plt.plot(X,y1,'r',label='Best Fit Line')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()

class Linear_Regression:

    def __init__(self):

        self.Xbar=0

        self.Ybar=0

        self.m=0

        self.b=0

    def fit (self,X,Y):

        self.X=X

        self.Y=Y

        self.Xbar=np.mean(self.X)

        self.Ybar=np.mean(self.Y)

        var1=(self.X-self.Xbar)*(self.Y-self.Ybar)

        var2=(self.X-self.Xbar)**2

        self.m=(sum(var1)/sum(var2))

        self.b=self.Ybar-(self.m*self.Xbar)

    

    def predict(self,X):

        return (self.m*X+self.b)
L=Linear_Regression()

L.fit(X,Y)

L.predict(4.8)