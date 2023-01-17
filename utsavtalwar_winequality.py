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
        df = pd.read_csv(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df.head()
df.isnull().sum()
df.quality.unique()
df['quality'].dtypes
df[['quality']] = df['quality'].apply(lambda x : (x - x) if int(x) < 6 else ((x - x) + 1))
df
#Splitting y and X
y = df['quality']
X = df.drop('quality', axis = 1)
y
#Splitting into test and train data
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.tree import DecisionTreeClassifier
d_model = DecisionTreeClassifier()
criterion = ['gini', 'entropy']
max_depth = [1, 3, 5, None]
splitter = ['best', 'random']
grid = GridSearchCV(estimator = d_model, cv= 3, param_grid=dict(criterion=criterion, max_depth=max_depth, splitter= splitter))
grid.fit(X_train, y_train)
print (grid.best_score_, grid.score(X_test, y_test))
from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter=10000)
l_model.fit(X_train, y_train)
l_model.score(X_test, y_test)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestClassifier(n_estimators, random_state=42))
    ])
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()
results = {}
for i in range(50,1000, 50):
    results[i] = get_score(i)
plt.plot(list(results.keys()), list(results.values()))
plt.show()
r_forest = RandomForestClassifier(n_estimators=400)
r_forest.fit(X_train, y_train)
r_forest.score(X_test, y_test)
y_valid = r_forest.predict(X_test)
y_valid
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_valid, y_test)
rfc_eval = cross_val_score(estimator = r_forest, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()