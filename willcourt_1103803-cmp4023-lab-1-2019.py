# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

import random

from sklearn import linear_model

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import Ridge

from yellowbrick.regressor import ResidualsPlot

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#test

#l = [random.randint(0,10) for i in range(50)]

#l
#rng = np.random.RandomState(500)



x1 = random.sample(range(500,2000), 50) #rng.rand(50)

x2 = random.sample(range(100,500), 50) # rng.rand(50)

#x3 = x1 * 3 + random.sample(range(500,2000), 50) #rng.randn(50)

random_vector = np.random.random(50)

x3 = x1 *  + random_vector

#Convert list to series

#y = [ ['x1'], ['x2'] ]

#df = pd.Series( (v[0] for v in y) )
y = []

count = 0

while count < 50:

    a = x3[count]

    b = x2[count]

    c = a + b

    y.append(c)

    count = count +1
y
[3,4] + [9,10]
x1
x2
x3
y
#len(y)

len(y)
df = {'X1 Values':x1, 'X2 Values':x2,'X3 Values':x3, 'Y Values':y}

dataf = pd.DataFrame(df)



dataf
dataf[['X1 Values','Y Values']].corr()

dataf[['X2 Values','Y Values']].corr()

dataf[['X3 Values','Y Values']].corr()

plt. scatter (x1,y)
plt.scatter(x2,y)
x_info = dataf[['X1 Values','X2 Values']]

y_info = dataf['Y Values']
regr = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x_info, y_info, test_size=0.30)
regr.fit(X_train,y_train)
regr.coef_
print("Regression Coefficients")

pd.DataFrame(regr.coef_,index=X_train.columns,columns=["Coefficient"])

model = Ridge()

visualizer = ResidualsPlot(model)



visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)

visualizer.score(X_test, y_test)



visualizer.show()

regr.intercept_
regr.score(X_test,y_test) #r-squared
predicted = regr.predict(X_test)
data2 = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})

data2



a=987

b=150

q = {'X':[a],'Y':[b]}

new=pd.DataFrame(q)

regr.predict(new)

x_info

mean_absolute_error(y_test, predicted)
mean_squared_error(y_test, predicted)
rootmeansqr = sqrt(mean_squared_error(y_test, predicted))

rootmeansqr


