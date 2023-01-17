# basic setup common to all analysis

import os
import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
color = sns.color_palette()

INPUT_DIR = '../input/' # location for Kaggle data files
print(os.listdir(INPUT_DIR))
# read the data - may take some time
iris = pd.read_csv(INPUT_DIR + 'Iris.csv')

# find out the shape of the data
print(iris.shape)
# what does the dataset contain
iris.sample(10)
# remove the id column - not required
iris = iris.drop('Id',axis=1)

# check data types, null-values, etc.
iris.info() 
sns.pairplot(iris, hue='Species')
from sklearn.model_selection import train_test_split
train, test = train_test_split(iris, test_size=0.35)

# features used for prediction - predictors - are conventionally denoted as X
X = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

X_train = train[X]
X_test = test[X]

# the feature to be predicted is conventionally denoted as y
y_train = train.Species
y_test = test.Species

X_train.sample(10)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# adapted from http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = ['KNN', 'Linear SVM', 'RBF SVM', 'Decision Tree', 'Random Forest', 'AdaBoost']
algos = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier()
]

results = ''
for name, algo in zip(names, algos):
    algo.fit(X_train_std, y_train)
    prediction = algo.predict(X_test_std)
    accuracy = metrics.accuracy_score(prediction, y_test)
    results += name + ': ' + str(accuracy) + "\n"

print(results)