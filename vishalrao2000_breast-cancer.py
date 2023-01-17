import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.drop('Unnamed: 32', inplace=True, axis=1)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
y = y.map({'M':0, 'B':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
