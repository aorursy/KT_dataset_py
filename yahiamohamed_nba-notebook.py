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
stats=pd.read_csv('../input/nba-regular-season-stats-20182019/nbastats2018-2019.csv')
X=stats.iloc[:,6].values

Y=stats.iloc[:,16].values



dict={stats.Name[i]:i for i in range(521)}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/5, random_state = 0)

plt.scatter(X_train, y_train)





plt.xlabel('FGA')

plt.ylabel('Points')

plt.show()
plt.scatter(X_test, y_test)

plt.xlabel('FGA')

plt.ylabel('Points')

plt.show()


X_train=X_train.reshape(416,1)

y_train=y_train.reshape(416,1)

X_train = np.c_[np.ones((X_train.shape[0])), X_train]

def gradientDescent(x, y, theta, alpha, m, numIterations):

    xTrans = x.transpose()

    for i in range(0, numIterations):

        hypothesis = np.dot(x, theta)

        loss = hypothesis - y

        

        cost = np.sum(loss ** 2) / (2 * m)

       

        

        gradient = np.dot(xTrans, loss) / m

       

        theta = theta - alpha * gradient

    return theta

import warnings

warnings.filterwarnings("ignore")

m=X_train.shape[0]

alpha=0.01

num=10000

theta=np.zeros((2,1))

theta2=gradientDescent(X_train,y_train,theta,alpha,m,num)



theta2
X_test=X_test.reshape(105,1)

y_test=y_test.reshape(105,1)

X_test = np.c_[np.ones((X_test.shape[0])), X_test]

print(theta2[0]+theta2[1]*X_test[50,1])

print(y_test[50])
