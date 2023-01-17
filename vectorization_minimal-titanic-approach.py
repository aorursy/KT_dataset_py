# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# -----------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------
# load data into pandas dataframe from .csv input flies provided by Kaggle
train = pd.read_csv("../input/train.csv") # load the train.csv data into the pandas dataframe called "train"
test = pd.read_csv("../input/test.csv") # load the test.csv data into the pandas dataframe called "test"
train_shape = train.shape # get the shape (columns and rows) of the training data 
train_rows = train.shape[0] # get the number of rows from index zero
train_cols = train.shape[1] # get the number of columns fro index one
print("Our training set has " + str(train_rows) + " rows")
print("Our taining set has " + str(train_cols) + " columns")
train.head() # display the first few rows
test_shape = test.shape
test_rows = test.shape[0]
test_cols = test.shape[1]
print("Our test set has " + str(test_rows) + " rows")
print("Our test set has " + str(test_cols) + " columns")
test.head()
# here we concatenate the two sets 
# we do this so that our data transformations span both sets correctly
combined = pd.concat((train,test)) # combine the train and test dataframes together

# just for testing combined.to_csv("combined.csv")

combined_shape = combined.shape # get the shape of the dataframe
combined_rows = combined.shape[0] 
combined_cols = combined.shape[1]
print("Our test set has " + str(combined_rows) + " rows")
print("Our test set has " + str(combined_cols) + " columns")
combined.head()

# now that sets are combined we'll remove columns we that won't help us make predictions
narrowed = combined.drop(['Cabin', 'Name', 'Survived', 'Ticket', 'Parch', 'PassengerId'], axis=1)
# now we use get_dummies to convert categorical data into one-hot columns
narrowed = pd.get_dummies(narrowed)
narrowed.head()
# now we fill in the blanks 
no_nulls = narrowed.fillna(narrowed.mean())
no_nulls.head()
# load up the matixes required for fitting
X_train = no_nulls[:train_rows]
print("X train: " + str(X_train.shape))
X_test = no_nulls[train_rows:]
print("X test: " + str(X_test.shape))
y = train.Survived
# load up classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),]

#for name, clf in zip(names, classifiers):
  #  clf.fit(X_train, y)
   # accuracy = round(clf.score(X_train, y) * 100, 2)
   # print(name, accuracy)
    


clf = RandomForestClassifier(max_depth=15, n_estimators=15, max_features=5)
clf.fit(X_train, y)
accuracy = round(clf.score(X_train, y) * 100, 2)
print(accuracy)
y.describe(include = 'all')  # Pandas describe function, wow!
predictions = clf.predict(X_test)
solution = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":predictions})
solution.to_csv("best_fit.csv", index = False)