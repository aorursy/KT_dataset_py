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



%matplotlib inline

import matplotlib.pyplot as plt

import scipy.optimize as opt

import sklearn.linear_model

import sklearn.model_selection

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing
df_train = pd.read_csv('/kaggle/input/hse-aml-2020/books_train.csv')

df_test = pd.read_csv('/kaggle/input/hse-aml-2020/books_test.csv')



df_train.columns = df_train.columns.str.replace(' ', '')

df_test.columns = df_test.columns.str.replace(' ', '')

#selecting columns

df_train = df_train[['bookID', 'title', 'authors', 'average_rating', 'language_code', 'num_pages', 'ratings_count', 'text_reviews_count', 'publication_date', 'publisher']]

df_test = df_test[['bookID', 'title', 'authors', 'language_code', 'num_pages', 'ratings_count', 'text_reviews_count', 'publication_date', 'publisher']]
le = preprocessing.LabelEncoder()

df_train['title'] = le.fit_transform(df_train['title'])

df_train['authors'] = le.fit_transform(df_train['authors'])

df_train['publisher'] = le.fit_transform(df_train['publisher'])



df_test['title'] = le.fit_transform(df_test['title'])

df_test['authors'] = le.fit_transform(df_test['authors'])

df_test['publisher'] = le.fit_transform(df_test['publisher'])

df_test
X = df_train.drop(['average_rating', 'language_code', 'publication_date'], axis = 1)

y = df_train['average_rating']



X_test = df_test.drop(['language_code', 'publication_date'], axis = 1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25)

regr = sklearn.linear_model.LinearRegression()

regr.fit(X_train, y_train)

prediction = regr.predict(X_valid)

test_prediction = regr.predict(X_test)

print(regr.score(X_valid, y_valid))
prediction
test_prediction
pred = pd.DataFrame({'bookID': X_test['bookID'].tolist(), 'average_rating': test_prediction.tolist()})

pred

pred.to_csv('submission_Lyakh.csv', index=False)