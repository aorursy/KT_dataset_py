# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/chess/games.csv')

df = data[data.increment_code == '15+15']
df.corr()
linear_reg = LinearRegression()



x = df.white_rating.values.reshape(-1,1)

y = df.black_rating.values.reshape(-1,1)



linear_reg.fit(x,y)



array = np.array([i for i in range(500,2500)]).reshape(-1,1)



b0 = linear_reg.intercept_

print('b0: ',b0)



b1 = linear_reg.coef_  #scikit  learn

print('b1: ',b1)



plt.scatter(df.white_rating,df.black_rating)

plt.xlabel('white_rating')

plt.ylabel('black_rating')

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color = 'red')

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(array, y_head))

print('Mean Squared Error:', metrics.mean_squared_error(array, y_head))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(array, y_head)))
x = df.iloc[:,[9,11]]

y = df.opening_ply.values.reshape(-1,1)



m_l_r = LinearRegression()

m_l_r.fit(x,y)



print('b0: ', m_l_r.intercept_)



print('b1,b2: ', m_l_r.coef_)
y = df.turns.values.reshape(-1,1)

x = df.opening_ply.values.reshape(-1,1)



plt.scatter(x,y)

plt.ylabel('turns')

plt.xlabel('opening_ply')


