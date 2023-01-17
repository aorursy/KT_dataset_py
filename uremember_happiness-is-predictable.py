# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
hap = pd.read_csv('../input/2017.csv')

hap.columns
hap = hap.drop(['Happiness.Rank','Whisker.high','Whisker.low'], axis=1)

hap.head()
g = sns.PairGrid(hap)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)
corr  = hap.corr()

fig, ax = plt.subplots(figsize = (12,6))

heat = sns.heatmap(corr, cmap="YlGnBu", annot=True, ax = ax )
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression
X = hap.drop(['Happiness.Score', 'Country'], axis=1)

y = hap['Happiness.Score']
X_train, X_test, y_train, y_test = train_test_split(X, y)
lm = LinearRegression()

svr = SVR()
lm.fit(X_train, y_train)

svr.fit(X_train, y_train)
lm_pred = lm.predict(X_test)

svr_pred = svr.predict(X_test)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (16, 8))

ax[0].plot(y_test, y_test-lm_pred,  'b+', label = 'linear')

ax[0].plot(y_test, y_test-svr_pred, 'r^', label = 'svr')

ax[0].set(xlabel= 'Happiness.Score/y_test', ylabel = 'Erro')

ax[1].set(xlabel= 'Happiness.Score/y_test', ylabel = 'Happiness.Score/Predicted')

ax[1].plot(y_test, lm_pred,  'b+', markersize=20, label = 'linear')

ax[1].plot(y_test, svr_pred, 'r^',label = 'svr' )

plt.legend(loc = 0)