# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

kc_house_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv", index_col = "id")
kc_house_data.head()
from xgboost import XGBRegressor
#prepare for predictors data and target data

X = kc_house_data.copy()

X.dropna(axis = 0, subset = ["price"], inplace=True)

y = X["price"]

X.drop(["price"], axis = 1, inplace=True)
X.shape
X_full, X_test, y_full, y_test = train_test_split(X, y, train_size=0.8, test_size = 0.2, random_state = 1)
print(X_full.shape)

print(X_full.columns)
for i in X_full.columns:

    print(i, X_full[i].dtype)
#change the date type of X_full["date"] from object to date object

for i in range(X_full.shape[0]):

    X_full["date"].iloc[i] = X_full["date"].iloc[i][0:8]

# print(pd.to_datetime(X_full["date"].iloc[0:5],format='%Y%m%d', errors='ignore'))
X_full["date"] = pd.to_datetime(X_full["date"], format = "%Y%m%d", errors = "ignore")
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(16,6))

sns.lineplot(x = X_full["date"], y=y_full)
cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()]

print(cols_with_missing)
print(X_full.columns)

X_full.head()
sns.scatterplot(x=X_full["zipcode"], y=y_full)
X_full_nodate = X_full.copy()

X_full_nodate = X_full_nodate.drop("date", axis=1)

print(["{}:{}".format(col,X_full_nodate[col].dtype) for col in X_full_nodate.columns])

X_full_nodate.head()
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error

X_train, X_valid, y_train, y_valid = train_test_split(X_full_nodate, y_full, train_size=0.8, test_size = 0.2, random_state = 0)

model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

model.fit(X_train, y_train, early_stopping_rounds=15, eval_set=[(X_valid, y_valid)], verbose=False)

preds = model.predict(X_valid)

score = mean_absolute_error(y_valid, preds)
print(score)
def gridsearch(n=1000, l=0.05):

    model = XGBRegressor(n_estimators=n, learning_rate = l, n_jobs=4)

    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_valid, y_valid)], verbose=False)

    preds = model.predict(X_valid)

    score = mean_absolute_error(y_valid, preds)

    return score



scores =[]

for i in [1000, 1100, 1200, 1300, 1400, 1500]:

    score = gridsearch(n=i)

    scores.append([i, score])

for i in scores:

    print("{}:{}".format(i[0], i[1]))
best = [1200, 0.05] #[n_estimeter, learning_rate]

model = XGBRegressor(n_estimators=best[0], learning_rate=best[1], n_jobs=4)

model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_valid, y_valid)], verbose=False)

X_test_copy = X_test.copy()

X_test_copy = X_test_copy.drop("date", axis=1)

preds=model.predict(X_test_copy)

print(mean_absolute_error(y_test, preds))