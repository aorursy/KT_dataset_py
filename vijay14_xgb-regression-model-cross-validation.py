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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
# read the data

data = pd.read_csv("/kaggle/input/tvradionewspaperadvertising/Advertising.csv")

data.shape, data.head()
data.isnull().any()
# plot TV vs Sales

sns.lineplot(x='Sales', y='TV', data=data, markers='*')

sns.lineplot(x='Sales', y='Radio',data=data, markers='+')

sns.lineplot(x='Sales', y='Newspaper',data=data, markers='-')
# plot TV advertosement vers sales

sns.lineplot(x='TV', y='Sales', data=data, markers=True)
# plot Radio advertosement vers sales

sns.lineplot(x='Radio', y='Sales', data=data, markers=True)
# plot Newspaper advertosement vers sales

sns.lineplot(x='Newspaper', y='Sales', data=data, markers=True)
# regression model to predict sales based on the advertisements

from xgboost import XGBRegressor

from sklearn.metrics import explained_variance_score

from sklearn.model_selection import train_test_split, KFold
xtrain, xtest, ytrain, ytest = train_test_split(data[['TV', 'Radio','Newspaper']], data.Sales, test_size=0.2)
model = XGBRegressor(max_depth=8,

                    n_estimators=200,

                    learning_rate=0.05)

model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)
explained_variance_score(y_pred=y_pred, y_true=ytest)
xtest1 = xtest.copy()

xtest1['y_pred'] = y_pred

xtest1['y'] = ytest

xtest1.head()
# k-fold crossvalidation

kf = KFold(n_splits=5, shuffle=True)

features = ['TV', 'Radio','Newspaper']

score = []

for train_ix, test_ix in kf.split(data):

    # print(train_ix, test_ix)

    train, test = data.loc[train_ix], data.loc[test_ix]

    model = XGBRegressor(max_depth=8,

                    n_estimators=200,

                    learning_rate=0.1)

    model.fit(train[features], train.Sales)

    y_pred = model.predict(test[features])

    scr = explained_variance_score(y_pred=y_pred, y_true=test.Sales)

    score.append(scr)

    print("Explained variance score: ", scr)
# Mean score

import numpy as np

np.mean(score)