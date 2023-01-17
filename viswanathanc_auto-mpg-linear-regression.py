# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/autompg-dataset/auto-mpg.csv")

data.head()
data.describe()
data.shape
data.horsepower
sum(data.horsepower=='?')
data=data[data.horsepower!='?']
data.horsepower=data.horsepower.astype('int64')
data.describe()
data['car name'].value_counts()
data['car name'].fillna('dddddd')
data['car name']=[i[0] for i in data['car name'].str.split(' ')]
data['car name'].unique()
data['car name']=data['car name'].replace(['chevrolet','chevy','chevroelt'],'chevrolet')

data['car name']=data['car name'].replace(['volkswagen','vw','vokswagen'],'volkswagen')

data['car name']=data['car name'].replace('maxda','mazda')

data['car name']=data['car name'].replace('toyouta','toyota')

data['car name']=data['car name'].replace('mercedes','mercedes-benz')

data['car name']=data['car name'].replace('nissan','datsun')

data['car name']=data['car name'].replace('capri','ford')
len(data['car name'])
data.info()
plt.hist(data.mpg)
sns.pairplot(data)
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
data=pd.concat([data,pd.get_dummies(data.origin,prefix='origin')],axis=1)

data.drop('origin',axis=1,inplace=True)
data=pd.concat([data,pd.get_dummies(data.cylinders,prefix='cylinders')],axis=1)

data.drop('cylinders',axis=1,inplace=True)
data=pd.concat([data,pd.get_dummies(data['model year'],prefix='year')],axis=1)

data.drop('model year',axis=1,inplace=True)
data.head(7)
data[['displacement','horsepower','weight','acceleration']]=StandardScaler().fit_transform(data[['displacement','horsepower','weight','acceleration']])
data=pd.concat([data,pd.get_dummies(data['car name'],prefix='car')],axis=1)

data.drop('car name',axis=1,inplace=True)
data.shape
y = data.pop('mpg')

X = data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
lr=LinearRegression()

lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

mean_squared_error(y_pred,y_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)