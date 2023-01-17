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
confirmed_global = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

confirmed_global.head(5)


confirmed_alberta = confirmed_global.loc[confirmed_global['Province/State']=='Alberta'].iloc[:,4:].values

# various usefful sklearn stuff
# classes we could use to fit regression models
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error


# data cleanup
con_on = None 
for i in confirmed_alberta:
    con_on=[k for k in i] 
# reshape data for sklearn
X = np.array(range(0, len(con_on))).reshape(-1, 1)
y = con_on

confirmed_alberta.shape

import matplotlib.pyplot as plt

svrrbf = SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1)
svrlin = SVR(kernel='linear', C=1, gamma='auto')

svrs = [svrrbf, svrlin]
kernellabel = ['RBF', 'Linear']
modelcolor = ['r', 'g']
lw = 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=modelcolor[ix],
                  label='{} model'.format(kernellabel[ix]))
    axes[ix].scatter(X, y, label='Confirmed Cases in Alberta')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].set_ylim((-10, max(y)+10))   # set the xlim to left, right
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("SVR", fontsize=12)
plt.show()


import matplotlib.pyplot as plt

svrrbf = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)
svrlin = SVR(kernel='linear', C=10, gamma='auto')

svrs = [svrrbf, svrlin]
kernellabel = ['RBF', 'Linear']
modelcolor = ['r', 'g']
lw = 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=modelcolor[ix],
                  label='{} model'.format(kernellabel[ix]))
    axes[ix].scatter(X, y, label='Confirmed Cases in Alberta')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].set_ylim((-10, max(y)+10))   # set the xlim to left, right
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("SVR", fontsize=12)
plt.show()




import matplotlib.pyplot as plt

svrrbf = SVR(kernel='rbf', C=50, gamma=0.1, epsilon=.1)
svrlin = SVR(kernel='linear', C=50, gamma='auto')

svrs = [svrrbf, svrlin]
kernellabel = ['RBF', 'Linear']
modelcolor = ['r', 'g']
lw = 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=modelcolor[ix],
                  label='{} model'.format(kernellabel[ix]))
    axes[ix].scatter(X, y, label='Confirmed Cases in Alberta')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].set_ylim((-10, max(y)+10))   # set the xlim to left, right
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("SVR", fontsize=12)
plt.show()



import matplotlib.pyplot as plt

svrrbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svrlin = SVR(kernel='linear', C=100, gamma='auto')

svrs = [svrrbf, svrlin]
kernellabel = ['RBF', 'Linear']
modelcolor = ['r', 'g']
lw = 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=modelcolor[ix],
                  label='{} model'.format(kernellabel[ix]))
    axes[ix].scatter(X, y, label='Confirmed Cases in Alberta')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].set_ylim((-10, max(y)+10))   # set the xlim to left, right
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("SVR", fontsize=12)
plt.show()



import matplotlib.pyplot as plt

svrrbf = SVR(kernel='rbf', C=10, gamma=0.5, epsilon=.1)
svrlin = SVR(kernel='linear', C=10, gamma=0.5)

svrs = [svrrbf, svrlin]
kernellabel = ['RBF', 'Linear']
modelcolor = ['r', 'g']
lw = 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=modelcolor[ix],
                  label='{} model'.format(kernellabel[ix]))
    axes[ix].scatter(X, y, label='Confirmed Cases in Alberta')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].set_ylim((-10, max(y)+10))   # set the xlim to left, right
fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("SVR", fontsize=12)
plt.show()


