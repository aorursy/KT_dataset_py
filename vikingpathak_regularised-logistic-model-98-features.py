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
df = pd.read_csv('/kaggle/input/etp-outcomebased-covid/Test_Covid.csv')
df.head()
print("Shape:", df.shape)
print(f"Null values present: {any(df.isnull().sum())}")
df.drop('Unnamed: 0', inplace=True, axis=1)
df.describe().T
# setting up X and y for training
y = df.Y
X = df.drop('Y', axis = 1)
from sklearn.linear_model import LogisticRegression
# Normalize X to speed up convergence
X /= X.max()
log_res = clf = LogisticRegression(penalty='l1',
                                   solver='liblinear',
                                   tol=1e-6,
                                   max_iter=int(1e6),
                                   warm_start=True,
                                   intercept_scaling=100,
                                   C=5,
                                   fit_intercept=True)
log_res.fit(X,y)
from sklearn.feature_selection import SelectFromModel

l1_sel_features = SelectFromModel(log_res, prefit=True)
for features in X.columns[l1_sel_features.get_support()]:
    print(features)
X_sel = X[X.columns[l1_sel_features.get_support()]]
def build_model(X, y):

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    print("------------------ Unregularised Model -----------------")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("------------------- Regularised Model -------------------")
    lr = LogisticRegression()
    X_sel = X_train[X_train.columns[l1_sel_features.get_support()]]
    lr.fit(X_sel, y_train)
    y_pred = lr.predict(X_test[X_test.columns[l1_sel_features.get_support()]])
    print(classification_report(y_test, y_pred))
build_model(X, y)