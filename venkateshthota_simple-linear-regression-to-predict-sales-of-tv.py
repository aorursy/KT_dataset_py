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
import pandas as pd

advertising=pd.read_csv("../input/tv-marketing/tvmarketing.csv")
import seaborn as sns
%matplotlib inline
sns.pairplot(advertising,x_vars=["TV"],y_vars=["Sales"],height=7, aspect=0.7,kind="scatter")
x=advertising["TV"]

x.head()
y=advertising["Sales"]

y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=100)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
import numpy as np

X_train=X_train[:,np.newaxis]

X_test=X_test[:,np.newaxis]
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)
print(lr.intercept_)

print(lr.coef_)
y_pred=lr.predict(X_test)
type(y_pred)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')