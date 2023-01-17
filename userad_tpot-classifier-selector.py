import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



data = pd.read_csv('../input/creditcard.csv')
Y = data["Class"]

X = data.copy().drop(["Time", "Amount", "Class"], 1)
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, subsample=0.01, n_jobs=1)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
#tpot.export('tpot_pipeline.py')



from tpot.export_utils import export_pipeline

print(export_pipeline(tpot._optimized_pipeline, tpot.operators, tpot._pset))
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC



training_features, testing_features, training_target, testing_target = train_test_split(X, Y, random_state=42)



exported_pipeline = LinearSVC(C=0.1, loss="hinge", penalty="l2")



exported_pipeline.fit(training_features, training_target)

results = exported_pipeline.predict(testing_features)
from sklearn.metrics import classification_report

print(classification_report(testing_target, results))