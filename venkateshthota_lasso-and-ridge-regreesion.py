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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.datasets import load_boston
df=load_boston()

df
dataset=pd.DataFrame(df.data)
dataset.head()
dataset.columns=df.feature_names

dataset.head()
df.target.shape
dataset["Price"]=df.target

dataset.head()
X=dataset.iloc[:,:-1] ## independent features

y=dataset.iloc[:,-1] ## dependent features
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
lin=LinearRegression()

mse=cross_val_score(lin,X,y,scoring="neg_mean_squared_error",cv=5)

mean_mse=np.mean(mse)

print(mean_mse)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_reg=GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error",cv=10)

ridge_reg.fit(X,y)
print(ridge_reg.best_params_)

print(ridge_reg.best_score_)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)



lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

prediction_lasso=lasso_regressor.predict(X_test)

prediction_ridge=ridge_reg.predict(X_test)
print(prediction_lasso)

print(prediction_ridge)
import seaborn as sns

sns.distplot(y_test-prediction_lasso)
import seaborn as sns

sns.distplot(y_test-prediction_ridge)