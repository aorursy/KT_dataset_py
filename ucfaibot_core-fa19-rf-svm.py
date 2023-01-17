from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-fa19-rf-svm").exists():

    DATA_DIR /= "ucfai-core-fa19-rf-svm"

elif DATA_DIR.exists():

    # no-op to keep the proper data path for Kaggle

    pass

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-rf-svm/data

    DATA_DIR = Path("data")
# Importing the important stuff

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.datasets import load_iris



iris = load_iris()

split = train_test_split(iris.data, iris.target, test_size=0.1)

X_train, X_test, Y_train, Y_test = split
trees = RandomForestClassifier(n_estimators=150, n_jobs=-1)

trees.fit(X_train, Y_train)
predictions = trees.predict(X_test)

confusion_matrix(Y_test, predictions)
print(matthews_corrcoef(Y_test, predictions))
# SVM regularization parameter, we'll keep it simple for now

C = 1.0 



models = (svm.SVC(kernel='linear', C=C),

          svm.SVC(kernel='poly', degree=3, C=C),

          svm.SVC(kernel='rbf', gamma=0.7, C=C))
models = (clf.fit(X_train, Y_train) for clf in models)
for clf in models:

    predictions = clf.predict(X_test)

    print(confusion_matrix(Y_test, predictions))
for clf in models:

    predictions = clf.predict(X_test)

    print(matthews_corrcoef(Y_test, predictions))
data = pd.read_csv(DATA_DIR / "train.csv")

data.info()
data.describe()
X = data.drop(columns='Class', axis=1)

Y = data['Class']



# sklearn requires a shape with dimensions (N, 1), 

# so we expand dimensions of x and y to put a 1 in the second dimension

print(f"X shape: {X.shape} Y shape: {Y.shape}")
# Make the magic happen!

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

X_test = pd.read_csv(DATA_DIR / "test.csv")



# to expedite things: pass `n_jobs=-1` so you can run across all available CPUs

# YOUR CODE HERE

raise NotImplementedError()
predictions = pd.DataFrame({'Id': Y_test.index, 'Class': predictions})



predictions.to_csv('submission.csv', header=['Id', 'Class'], index=False)