import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# Podemos observar as primeiras linhas dele.

#titanic_df.head()

test_df.head()
numeric_features = ['Pclass', 'SibSp', 'Parch', 'Fare']

titanic_df[numeric_features].tail(10)
train_X = titanic_df[numeric_features].as_matrix()

print(train_X.shape)

train_y = titanic_df['Survived'].as_matrix()

print(train_y.shape)
train_X
train_y
import seaborn as sns

sns.countplot(titanic_df['Survived']);
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(train_X, train_y)
dummy_clf.score(train_X, train_y)
train_X[-10:]
train_y[-10:]
dummy_clf.predict(train_X[-10:])
test_df.head()
test_df['Fare'] = test_df['Fare'].fillna(0)
test_X = test_df[numeric_features].as_matrix()

print(test_X.shape)
test_X
y_pred = dummy_clf.predict(test_X)
y_pred
sample_submission_df = pd.DataFrame()
sample_submission_df['PassengerId'] = test_df['PassengerId']

sample_submission_df['Survived'] = y_pred

sample_submission_df
sample_submission_df.to_csv('basic_dummy_classifier.csv', index=False)