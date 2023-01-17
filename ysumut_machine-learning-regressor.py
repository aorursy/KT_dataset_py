# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')

data.head(10)
pelvic = data.iloc[:,0].values.reshape(-1,1)

sacral = data.iloc[:,3].values
plt.scatter(pelvic,sacral)

plt.xlabel('pelvic')

plt.ylabel('sacral')

plt.grid(True)

plt.show()
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(pelvic,sacral)



print( lr.predict([[60]]) )



print( r2_score(sacral, lr.predict(pelvic)) )
x_ = np.arange(min(pelvic), max(pelvic), 0.01).reshape(-1,1)



plt.plot(x_, lr.predict(x_), c='red', label='linear model')

plt.scatter(pelvic, sacral, c='blue', alpha=0.5, label='normal data')



plt.xlabel('pelvic')

plt.ylabel('sacral')

plt.grid(True)

plt.legend()

plt.show()
from sklearn.tree import DecisionTreeRegressor



dt = DecisionTreeRegressor()

dt.fit(pelvic, sacral)



print( dt.predict([[60]]) )



print( r2_score(sacral, dt.predict(pelvic)) )
x_ = np.arange(min(pelvic), max(pelvic), 0.01).reshape(-1,1)



plt.plot(x_, dt.predict(x_), c='green', label='decision tree')

plt.scatter(pelvic, sacral, c='blue', alpha=0.2, label='normal data')



plt.xlabel('pelvic')

plt.ylabel('sacral')

plt.grid(True)

plt.legend()

plt.show()
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(pelvic,sacral)



print( rf.predict([[60]]) )



print( r2_score(sacral, rf.predict(pelvic)) )
x_ = np.arange(min(pelvic), max(pelvic), 0.01).reshape(-1,1)



plt.plot(x_, rf.predict(x_), c='green', label='random forest')

plt.scatter(pelvic, sacral, c='blue', alpha=0.2, label='normal data')



plt.xlabel('pelvic')

plt.ylabel('sacral')

plt.grid(True)

plt.legend()

plt.show()