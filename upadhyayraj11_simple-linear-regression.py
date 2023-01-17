# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')
df_train.head(3)
df_test.head(3)
print("max value in column x =",df_train['x'].max())

print("max value in column y =",df_train['y'].max())
np.where(np.isnan(df_train['x']))
np.where(np.isnan(df_train['y']))
'''so here we find that at row 213 we are having NaN value so there are two ways to tackle this situation.

 one you can discard that row 

 second add mean or median value of that column at that specific spot.

 but before that let us plot graph between x and y.'''
#now let's plot graph (x vs y)

plt.title("Simple Linear Regression Example")

plt.xlabel("X values")

plt.ylabel("Y values")

plt.scatter(df_train.x,df_train.y,color='blue')
value = df_train.x.mean()

df_train.y = df_train.y.fillna(value)

df_train.head(3)
#now let's plot graph (x vs y)

plt.title("Simple Linear Regression Example")

plt.xlabel("X values")

plt.ylabel("Y values")

plt.scatter(df_train.x,df_train.y,color='blue')
df_train = pd.read_csv('../input/train.csv')
np.where(np.isnan(df_train['y']))

df_train = df_train.drop(213)
#now let's plot graph (x vs y)

plt.title("Simple Linear Regression Example")

plt.xlabel("X values")

plt.ylabel("Y values")

plt.scatter(df_train.x,df_train.y,color='blue')
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df_train[['x']],df_train.y)
df_train.head(1)
model.predict([[24]])
model.score(df_test[['x']],df_test['y'])