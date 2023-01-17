import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os
df = pd.read_csv('../input/data.csv')

df.head()
df.columns
df.describe()
df.dtypes
data = df.drop(['Unnamed: 32', 'id'], axis=1)
data.head()
data.diagnosis.value_counts().plot(kind = 'pie', figsize=[10,10])
fig, axes = plt.subplots(15, ncols=2)

j = 15

features = X.columns

for i in range(15):

    X[features[i]].plot(ax=axes[i,0] ,kind = 'hist', figsize = (18,30))

    X[features[j]].plot(ax=axes[i,1], kind = 'hist')

    j = j+1
from scipy import stats

X_zscore = stats.zscore(X)

zscore_df = pd.DataFrame(X_zscore, columns=X.columns)

zscore_df.head()
X = data.drop('diagnosis', axis=1)

y = data.diagnosis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_df = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_df, columns=X.columns)
scaled_df.head()
scaled_df.shape
fig, axes = plt.subplots(15, ncols=2)

j = 15

features = X.columns

for i in range(15):

    scaled_df[features[i]].plot(ax=axes[i,0] ,kind = 'hist', figsize = (18,30))

    scaled_df[features[j]].plot(ax=axes[i,1], kind = 'hist')

    j = j+1
from category_encoders import OneHotEncoder

from category_encoders.binary import BinaryEncoder

one_hot = OneHotEncoder()

y_enoded = one_hot.fit_transform(list(y))

y_enoded.drop('0_-1', axis=1, inplace= True)
y_enoded.head()
from sklearn.model_selection import train_test_split

X_train, X_test,  y_train, y_test = train_test_split(scaled_df, y_enoded, test_size = .20)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

clf_tree = DecisionTreeClassifier()

help(clf_tree)
print(X_train.shape)

print(y_train.shape)



print(X_test.shape)

print(y_test.shape)
clf_tree.fit(X_train, y_train)

y_train_pred = clf_tree.predict(X_train)

y_test_pred = clf_tree.predict(X_test)

print("Train accuracy: ", accuracy_score(y_train, y_train_pred))

print("Test accuracy: ", accuracy_score(y_test, y_test_pred))
y_pred = d_tree.predict(scaled_df)
accuracy_score(y_enoded, y_pred)