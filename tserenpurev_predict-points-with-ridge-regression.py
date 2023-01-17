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
# Method of linear regression

# https://viblo.asia/p/logistic-regression-with-python-gDVK2wQrZLj



# Instruments

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import DictVectorizer

from scipy.sparse import hstack

from sklearn.linear_model import Ridge



import plotly.graph_objects as go

import matplotlib.pyplot as plt
# Data Load



data = pd.read_csv('/kaggle/input/wine-reviews/winemag-data_first150k.csv')

data = data[['description', 'province', 'winery', 'points']]

# Formatting



data['description'].str.lower()

data['description'].replace('[^a-zA-Z0-9]', ' ', regex = True)

data['winery'].fillna('nan', inplace=True)

data['province'].fillna('nan', inplace=True)
data_train, data_validate, data_test = np.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])



print(data.shape)

print(data_train.shape)

print(data_validate.shape)

print(data_test.shape)



vectorizer = TfidfVectorizer(min_df=5)

X_tfidf = vectorizer.fit_transform(data_train['description'])

X_test_tfidf = vectorizer.transform(data_test['description'])
enc = DictVectorizer()

X_train_categ = enc.fit_transform(data_train[['province', 'winery']].to_dict('records'))

X_test_categ = enc.transform(data_test[['province', 'winery']].to_dict('records'))

                                  

                                  

X = hstack([X_tfidf,X_train_categ])

X_test = hstack([X_test_tfidf,X_test_categ])

clf = Ridge(alpha=1.0, random_state=241)



y = data_train['points']



clf.fit(X, y) 
rslt = clf.predict(X_test)
scores = clf.score(X, y)

print(scores)
real_points = data_test.points.tolist()

predict_points = [int(rsl) for rsl in rslt]



fig = go.Figure()

# fig.add_trace(go.Scatter(y=real_points[:100], mode='lines+markers', name='real_points'))

# fig.add_trace(go.Scatter(y=predict_points[:100], mode='lines+markers', name='predict_points'))



fig = go.Figure(data=[

    go.Bar(name='Real', y=real_points[:100]),

    go.Bar(name='Predict', y=predict_points[:100])

])





fig.show()
clf2 = Ridge(alpha=1.0, random_state=241)

clf2.fit(X_tfidf, y)

rslt2 = clf2.predict(X_test_tfidf)
real_points2 = data_test.points.tolist()

predict_points2 = [int(rsl) for rsl in rslt2]
fig2 = go.Figure()

fig2 = go.Figure(data=[

    go.Bar(name='Real', y=real_points2[:100]),

    go.Bar(name='Predict', y=predict_points2[:100])

])

fig2.show()