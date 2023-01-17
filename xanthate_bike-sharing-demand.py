%matplotlib inline

import numpy as np

import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime
os.listdir('../input/')
PATH = '../input/train.csv'

train = pd.read_csv(PATH)

train.head()
train.describe()
train.isnull().sum()
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(train["count"], ax=ax, hist=True, kde=True)
# log transformation, add 1 because some values are close to 0

y = np.log1p(train["count"])
sns.distplot(y, kde=True)

plt.show()
fig, ax = plt.subplots(1, 4, figsize=(20, 4))

sns.distplot(train.temp, kde=True, color='green', ax=ax[0])

sns.distplot(train.atemp, kde=True, color='green', ax=ax[1])

sns.distplot(train.humidity, kde=True, color='green', ax=ax[2])

sns.distplot(train.windspeed, kde=True, color='green', ax=ax[3])

plt.show()
train['windspeed'].describe()
# datetime feature

train["date"] = train.datetime.apply(lambda x : x.split()[0])

train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

train["year"] = train.datetime.apply(lambda x : x.split()[0].split("-")[0])

train["weekday"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
train.head()
train_X = train.drop(['casual','registered','datetime', 'date', 'year', 'atemp'],axis=1)

train_y = train['count']

train_X.head()
train_X = train_X.drop(['count'], axis=1)
train_X = train_X.values
from sklearn.ensemble import RandomForestRegressor

'''

from sklearn.grid_search import GridSearchCV



estimators = list(range(50, 100, 10))

sam_leaf = list(range(90, 101, 10))

param_grid = dict(n_estimators=estimators, min_samples_leaf=sam_leaf)

print(param_grid)



rf = RandomForestRegressor()

grid = GridSearchCV(rf, param_grid, cv=10, scoring='neg_mean_squared_error')

grid.fit(train_X, train_y)

print(grid.best_score_)

print(grid.best_params_)

'''
model = RandomForestRegressor(n_estimators=120, min_samples_leaf=18)

model.fit(train_X, train_y)

test_file_path = '../input/test.csv'

test = pd.read_csv(test_file_path)

test.head()
test["date"] = test.datetime.apply(lambda x : x.split()[0])

test["hour"] = test.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

test["year"] = test.datetime.apply(lambda x : x.split()[0].split("-")[0])

test["weekday"] = test.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

test["month"] = test.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
test.head()
# Replace the zeros with mean

test["windspeed"] = test["windspeed"].replace(0.0, train['windspeed'].mean())
test_X = test.drop(['datetime', 'date', 'year', 'atemp'],axis=1)

test_X.head()
prediction = model.predict(test_X)

print(prediction)
sub = pd.DataFrame({'datetime': test.datetime, 'count': prediction})

sub.to_csv('submission.csv', index=False)