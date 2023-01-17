import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from sklearn.neural_network import MLPClassifier
import sklearn
import time
# Define dataset
df = pd.read_csv("../input/Iris.csv")
from sklearn.model_selection import train_test_split
df.head()
array = df.values
X = array[:,1:5]
Y = array[:,5]
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# How many different kinds of irises do we have in each bucket?
#df["Species"].value_counts()
df.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")

def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5)
# Decision tree
import sklearn.tree 
dtree = sklearn.tree.DecisionTreeClassifier(min_samples_split=2)
dtree.fit(X_train, Y_train)
print_accuracy(dtree.predict)

# explain all the predictions in the test set

# Logistic regression
regr = sklearn.linear_model.LogisticRegression()
regr.fit(X_train, Y_train)
print_accuracy(regr.predict)

# TODO: visualize outputs
# Neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(X_train, Y_train)
print_accuracy(nn.predict)

# TODO: make visualizations