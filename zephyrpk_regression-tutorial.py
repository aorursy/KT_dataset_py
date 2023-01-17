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
import numpy as np

import pandas as pd

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

from sklearn.datasets import load_boston

from sklearn.model_selection import cross_val_score, KFold

from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge
boston = load_boston()
X, y = boston['data'], boston['target']
print(boston.DESCR)
boston.feature_names
X[:2]
lasso = Lasso(alpha=0.1)

lasso.fit(X,y)

lasso.coef_
lasso = Lasso(alpha=10)

lasso.fit(X,y)

lasso.coef_
n_alphas = 200

alphas = np.linspace(0.1,10,n_alphas)

model = Lasso()

coefs = []

for a in alphas:

    model.set_params(alpha = a)

    model.fit(X, y)

    coefs.append(model.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Lasso coefficients as function of the regularization')

plt.axis('tight')

plt.show()
lasso_cv = LassoCV(alphas=alphas, cv=3, random_state=42)

lasso_cv.fit(X, y)
lasso_cv.coef_
lasso_cv.alpha_
cross_val_score(Lasso(lasso_cv.alpha_), X, y, cv=3,

               scoring='neg_mean_squared_error')
abs(cross_val_score(Lasso(lasso_cv.alpha_), X, y, cv=3,

               scoring='neg_mean_squared_error').mean())
lasso_cv.alphas[:10]
plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(1))

plt.axvline(lasso_cv.alpha_, c='g')
ridge_alphas = np.logspace(-2,6,n_alphas)
ridge_cv = RidgeCV(alphas=ridge_alphas,

                  scoring='neg_mean_squared_error',

                  cv=3)

ridge_cv.fit(X,y)
ridge_cv.alpha_
ridge_cv.coef_
n_alphas = 200

alphas = np.linspace(2,6,n_alphas)

model = Ridge()

coefs = []

for a in alphas:

    model.set_params(alpha = a)

    model.fit(X, y)

    coefs.append(model.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)

#ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as function of the regularization')

plt.axis('tight')

plt.show()