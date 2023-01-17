import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn

%matplotlib inline



print(np.__version__)

print(pd.__version__)

import sys

print(sys.version)

import sklearn

print(sklearn.__version__)
from sklearn import tree
X = [[0, 0], [1, 2]]

y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
clf.predict([[2., 2.]])
clf.predict_proba([[2. , 2.]])
clf.predict([[0.4, 1.2]])
clf.predict_proba([[0.4, 1.2]])
clf.predict_proba([[0, 0.2]])
from sklearn.datasets import load_iris

from sklearn import tree

iris = load_iris()
iris.data[0:5]
iris.feature_names
X = iris.data[:, 2:]
y = iris.target
y
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)
import numpy as np

import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt

%matplotlib inline
def gini(p):

    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):

    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):

    return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)



ent = [entropy(p) if p != 0 else None for p in x]



sc_ent = [e*0.5 if e else None for e in ent]

err = [error(i) for i in x]
fig = plt.figure(figsize=(10,8))

ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 

                   ['Entropy', 'Entropy (scaled)', 

                   'Gini Impurity', 

                   'Misclassification Error'],

                   ['-', '-', '--', '-.'],

                   ['black', 'lightgray',

                      'red', 'green', 'cyan']):

     line = ax.plot(x, i, label=lab, 

                    linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),

           ncol=3, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')

ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

plt.ylim([0, 1.1])

plt.xlabel('p(i=1)')

plt.ylabel('Impurity Index')

plt.show()
from sklearn import tree
X = [[0, 0], [3,3]]

y = [0.75, 3]
tree_reg = tree.DecisionTreeRegressor(random_state=42)
tree_reg = tree_reg.fit(X, y)
tree_reg.predict([[1.5, 1.5]])
# Import the necessary modules and libraries

import numpy as np

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



# Create a random dataset

rng = np.random.RandomState(1)

X = np.sort(5 * rng.rand(80, 1), axis=0)

y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))



# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X, y)

regr_2.fit(X, y)



# Predict

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)



# Plot the results

plt.figure(figsize=(10,8))

plt.scatter(X, y, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.plot(X_test, y_1, color="cornflowerblue",

         label="max_depth=2", linewidth=2)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()
# Import the necessary modules and libraries

import numpy as np

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



# Create a random dataset

rng = np.random.RandomState(1)

X = np.sort(5 * rng.rand(80, 1), axis=0)

y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))



# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)

regr_1.fit(X, y)

regr_2.fit(X, y)



# Predict

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)



# Plot the results

plt.figure(figsize=(10,8))

plt.scatter(X, y, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.plot(X_test, y_1, color="cornflowerblue",

         label="max_depth=2", linewidth=2)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()