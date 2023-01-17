# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head()

train.describe(include='all')
train['winPlacePerc']=pd.to_numeric(train['winPlacePerc'],errors = 'coerce')
train = train.drop(['Id','groupId','matchId'], axis=1)
train['matchType'].describe()
train['matchType'].unique()
train = pd.get_dummies(train,columns=['matchType'])
#matchType_disct = {'squad-fpp': 1, 'duo': 2, 'solo-fpp': 3, 'squad': 4, 'duo-fpp': 5, 'solo': 6,
 #      'normal-squad-fpp': 7, 'crashfpp': 8, 'flaretpp': 9, 'normal-solo-fpp': 10,
  #     'flarefpp': 11, 'normal-duo-fpp':12, 'normal-duo':13, 'normal-squad':14,
  #     'crashtpp': 15, 'normal-solo': 16}
#train['matchType'].replace(matchType_disct,inplace=True)
train.head()
train = train.dropna(how = 'any')
train[train.isnull().any(axis=1)]
from sklearn.model_selection import train_test_split
X= train.drop('winPlacePerc',axis= 1)
y= train['winPlacePerc']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20, random_state = 0)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
linear_model.score(X_train,y_train)
linear_model.coef_
prediction = X_train.columns
coef = pd.Series(linear_model.coef_,prediction).sort_values()
print(coef)
y_predict = linear_model.predict(X_test)
y_predict
%pylab inline
pylab.rcParams['figure.figsize']= (15,6)
plt.plot(y_predict, label= 'Predicted')
plt.plot(y_test.values, label= 'Actual')
plt.ylabel('winPlacePerc')
plt.legend()
plt.show()
r_square = linear_model.score(X_test,y_test)
r_square
from sklearn.metrics import mean_squared_error
linear_model_mse = mean_squared_error(y_predict,y_test)
linear_model_mse
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=5,normalize=True)
lasso_model.fit(X_train,y_train)
lasso_model.score(X_train,y_train)
coef = pd.Series(lasso_model.coef_,prediction).sort_values()
print(coef)
y_predict2 = lasso_model.predict(X_test)
y_predict2
%pylab inline
pylab.rcParams['figure.figsize']= (15,6)
plt.plot(y_predict2, label= 'Predicted')
plt.plot(y_test.values, label= 'Actual')
plt.ylabel('winPlacePerc')
plt.legend()
plt.show()