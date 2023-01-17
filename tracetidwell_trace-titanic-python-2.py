import matplotlib.pyplot as plt

%matplotlib inline

import random

import numpy as np

import pandas as pd

from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

import sklearn.ensemble as ske

import tensorflow as tf

from tensorflow.contrib import skflow
titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_df.head()
titanic_df['Survived'].mean()
titanic_df.groupby('Pclass').mean()
class_sex_grouping = titanic_df.groupby(['Pclass', 'Sex']).mean()

print(class_sex_grouping['Survived'])
class_sex_grouping['Survived'].plot.bar()
group_by_age = pd.cut(titanic_df['Age'], np.arange(0, 90, 10))

age_grouping = titanic_df.groupby(group_by_age).mean()

age_grouping['Survived'].plot.bar()
titanic_df.count()
titanic_df = titanic_df.drop(['Cabin'], axis = 1)
titanic_df = titanic_df.dropna()
titanic_df.count()
def preprocess_titanic_df(df) :

    processed_df = df.copy()

    le = preprocessing.LabelEncoder()

    processed_df.Sex = le.fit_transform(processed_df.Sex)

    processed_df.Embarked = le.fit_transform(processed_df.Embarked)

    processed_df = processed_df.drop(['Name', 'Ticket'], axis = 1)

    return processed_df
processed_df = preprocess_titanic_df(titanic_df)

processed_df.count()

processed_df
X = processed_df.drop(['Survived'], axis = 1).values

Y = processed_df['Survived'].values
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(x_train, y_train)

clf_dt.score(x_test, y_test)
shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)

def test_classifier(clf) :

    scores = cross_validation.cross_val_score(clf, X, Y, cv=shuffle_validator)

    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
test_classifier(clf_dt)
clf_rf = ske.RandomForestClassifier(n_estimators=50)

test_classifier(clf_rf)
clf_gb = ske.GradientBoostingClassifier(n_estimators=50)

test_classifier(clf_gb)
eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])

test_classifier(eclf)
def custom_model(X, Y) :

    layers = skflow.ops.dnn(X, [20, 40, 20], tf.tanh)

    return skflow.models.logistic_regression(layers, Y)
tf_clf_c = skflow.TensorFlowEstimator(model_fn=custom_model, n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)

tf_clf_c.fit(x_train, y_train)

metrics.accuracy_score(y_test, tf_clf_c.predict(x_test))