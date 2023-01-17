# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

from graphviz import Source

from sklearn.tree import export_graphviz



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "decision_trees"

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)



def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn import linear_model

 

# Data set

x = np.array(list(range(1, 11))).reshape(-1, 1)

y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).ravel()

 

# Fit regression model

model1 = DecisionTreeRegressor(max_depth=1)

model2 = DecisionTreeRegressor(max_depth=3)

model3 = linear_model.LinearRegression()

model1.fit(x, y)

model2.fit(x, y)

model3.fit(x, y)

 

# Predict

X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]

y_1 = model1.predict(X_test)

y_2 = model2.predict(X_test)

y_3 = model3.predict(X_test)

 

# Plot the results

plt.figure()

plt.scatter(x, y, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.plot(X_test, y_1, color="cornflowerblue",

         label="max_depth=1", linewidth=2)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=3", linewidth=2)

plt.plot(X_test, y_3, color='red', label='liner regression', linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()
export_graphviz(

        model1,

        out_file=os.path.join(IMAGES_PATH, "regression_tree_1.dot"),

        feature_names=["x1"],

        rounded=True,

        filled=True

    )
export_graphviz(

        model2,

        out_file=os.path.join(IMAGES_PATH, "regression_tree_2.dot"),

        feature_names=["x1"],

        rounded=True,

        filled=True

    )
Source.from_file(os.path.join(IMAGES_PATH, "regression_tree_1.dot"))
Source.from_file(os.path.join(IMAGES_PATH, "regression_tree_2.dot"))
test = model2.predict([[7]])

print(test)
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier



iris = load_iris()

X = iris.data[:, 2:] # petal length and width

y = iris.target



tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)

tree_clf.fit(X, y)
from graphviz import Source

from sklearn.tree import export_graphviz



export_graphviz(

        tree_clf,

        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),

        feature_names=iris.feature_names[2:],

        class_names=iris.target_names,

        rounded=True,

        filled=True

    )



Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
from matplotlib.colors import ListedColormap



def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):

    x1s = np.linspace(axes[0], axes[1], 100)

    x2s = np.linspace(axes[2], axes[3], 100)

    x1, x2 = np.meshgrid(x1s, x2s)

    X_new = np.c_[x1.ravel(), x2.ravel()]

    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    if not iris:

        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])

        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    if plot_training:

        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")

        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")

        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")

        plt.axis(axes)

    if iris:

        plt.xlabel("Petal length", fontsize=14)

        plt.ylabel("Petal width", fontsize=14)

    else:

        plt.xlabel(r"$x_1$", fontsize=18)

        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

    if legend:

        plt.legend(loc="lower right", fontsize=14)



plt.figure(figsize=(8, 4))

plot_decision_boundary(tree_clf, X, y)

plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)

plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)

plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)

plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)

plt.text(1.40, 1.0, "Depth=0", fontsize=15)

plt.text(3.2, 1.80, "Depth=1", fontsize=13)

plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)



save_fig("decision_tree_decision_boundaries_plot")

plt.show()
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])