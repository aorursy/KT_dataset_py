# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Ridge , Lasso

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split , cross_val_score







from sklearn import model_selection

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV,LassoCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/lasso.csv")
df.head()
df.dropna()
dms = pd.get_dummies(data[['League','Division','NewLeague']])

y=data["Salary"].astype('float64')
x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')
X = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1).astype("float64")

df = pd.DataFrame(data=data, columns=['Salary'])

df = df.fillna(0)
y=df.Salary.astype(int)



y
x_train,x_test,y_train,y_test = train_test_split(x_, y, test_size=0.25 , random_state=42)
lasso_model=Lasso().fit(x_train,y_train)
lasso_model
lasso_model.intercept_
lasso_model.coef_
lamdalar = 10**np.linspace(10,-2,100)*0.5



coefs = []

for a in lamdalar:

    lasso_model.set_params(lamdalar = a)

    lasso_model.fit(x_train,y_train)

    coefs.append(lasso.coef_)
ax = plt.gca()

ax.plot(lamdalar,coefs)

ax.set_xscale("log")
lasso_model.predict(x_train)[0:5]
lasso_model.predict(x_test)[0:5]
y_pred=lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
lasso_cv_model = LassoCV(alphas =lamdalar ,cv=10 , max_iter=10000).fit(x_train,y_train)
lasso_cv_model.alpha_
lasso_tum = Lasso().set_params(alpha= lasso_cv_model.alpha_).fit(x_train,y_train)
y_pred=lasso_tum.predict(x_test)

np.sqrt(mean_squared_error(y_test,y_pred))