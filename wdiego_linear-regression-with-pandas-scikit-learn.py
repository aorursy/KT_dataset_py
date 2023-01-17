# libraries import

import pandas as pd

import numpy as np

from scipy import stats

from datetime import datetime

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline
# loading the data

df = pd.read_csv("../input/building1retail.csv", index_col='Timestamp', date_parser=lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))



# show first 5 rows

df.head()
df.shape
# show column types

df.dtypes
# exploring the data

df.plot(figsize=(18,5))
# check if there is no missing values

df.isnull().values.any()
# histogram of the data

df.hist()
# filter records that are greater than 3 std, to remove outliers

df = df[(np.abs(stats.zscore(df)) < 3.).all(axis=1)]

df.shape
# graph without outliers

df.plot(figsize=(18,5))
# scatter plot to see there are linear relationship

plt.scatter(df['OAT (F)'], df['Power (kW)'])
# checking timezone on a daytime per column

df.loc['2010-01-01', ['OAT (F)']].plot()
# checking timezone on a daytime per column

df.loc['2010-01-01', ['Power (kW)']].plot()
# linear regression model

X = pd.DataFrame(df['OAT (F)'])

y = pd.DataFrame(df['Power (kW)'])

model = LinearRegression()

scores = []



# split the records into 3 folds and train 3 times the model, 

# test and get the score of each training

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X, y)):

  model.fit(X.iloc[train,:], y.iloc[train,:])

  score = model.score(X.iloc[test,:], y.iloc[test,:])

  scores.append(score)

print(scores)
# To archieve a better model, let's consider the hour of the day

X['tod'] = X.index.hour

# drop_first = True removes multi-collinearity

add_var = pd.get_dummies(X['tod'], prefix='tod', drop_first=True)

# Add all the columns to the model data

X = X.join(add_var)

# Drop the original column that was expanded

X.drop(columns=['tod'], inplace=True)

print(X.head())
# training again with the new dummie columns

model = LinearRegression()

scores = []

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X, y)):

 model.fit(X.iloc[train,:], y.iloc[train,:])

 scores.append(model.score(X.iloc[test,:], y.iloc[test,:]))

print(scores)


