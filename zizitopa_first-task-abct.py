%matplotlib inline
from sklearn.datasets import make_moons

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4)
plt.scatter(X_moons[:,0], X_moons[:,1],  c=y_moons, alpha=0.9)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_moons, y_moons, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
decision_tree = DecisionTreeClassifier()

decision_tree.get_params()
parameters = {

    'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 75, 100]    

}
clf = GridSearchCV(decision_tree, parameters)
%%time

clf.fit(X_train, Y_train)
res = (

    pd.DataFrame({

        "mean_test_score": clf.cv_results_["mean_test_score"],

        "mean_fit_time": clf.cv_results_["mean_fit_time"]})

      .join(pd.json_normalize(clf.cv_results_["params"]).add_prefix("param_"))

)

res
new_dt = DecisionTreeClassifier(max_leaf_nodes=5)
new_dt.fit(X_train, Y_train)
new_dt.score(X_test, Y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(new_dt.predict(X_test), Y_test)