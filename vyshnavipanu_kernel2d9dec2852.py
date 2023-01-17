# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import  load_boston
data=load_boston()
data1=pd.DataFrame(data.data,columns=data.feature_names)
data1.head()
data1.info()
data1.describe()
data1['medv']=data.target
data1.head()
import matplotlib.pyplot as plt

import seaborn as sns
data1.head()
from sklearn.model_selection import train_test_split

x=data1.drop(['medv'],axis=1)

y=data1.medv

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

y_train_pred=lr.predict(x_train)

y_test_pred=lr.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error

r2_score(y_train,y_train_pred)
r2_score(y_test,y_test_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
print(rmse)
plt.scatter(y_test,y_test_pred)

plt.title('actual predicted')
from sklearn.tree import DecisionTreeRegressor

DTR=DecisionTreeRegressor()

DTR.fit(x_train,y_train)

y_train_pred=DTR.predict(x_train)

y_test_pred=DTR.predict(x_test)
r2_score(y_test_pred,y_test)
r2_score(y_train_pred,y_train)
from sklearn.preprocessing import StandardScaler



ssX = StandardScaler()

ssy = StandardScaler()

X_scaled = ssX.fit_transform(x_train)

#y_scaled = ssy.fit_transform(y_train.reshape(1,-1))