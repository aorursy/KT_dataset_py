# Load the MNIST dataset

import numpy as np



try:

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1)

    mnist.target = mnist.target.astype(np.int64)

except ImportError:

    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata('MNIST original')
# Split 50,000 instances for training, 10,000 for validation, and 10,000 for testing.



from sklearn.model_selection import train_test_split



X_train_val, X_test, y_train_val, y_test = train_test_split( mnist.data, mnist.target, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split( X_train_val, y_train_val, test_size=10000, random_state=42)
# Train Random Forest classifier, Extra-Trees classifier, SVM and MLP



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier



random_forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

extra_trees_clf = ExtraTreesClassifier(n_estimators=10, random_state=42)

svm_clf = LinearSVC(random_state=42)

mlp_clf = MLPClassifier(random_state=42)



estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]

for estimator in estimators:

    estimator.fit(X_train, y_train)
# .score() method directly calls sklearn.metrics.accuracy_score method.



[estimator.score(X_val, y_val) for estimator in estimators]
# Combine the classifiers into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier.



from sklearn.ensemble import VotingClassifier



named_estimators = [ ("random_forest_clf", random_forest_clf), ("extra_trees_clf", extra_trees_clf), ("svm_clf", svm_clf), ("mlp_clf", mlp_clf)]
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
voting_clf.score(X_val, y_val)
[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
# remove an estimator by setting it to None using set_params()



voting_clf.set_params(svm_clf=None)
# Updated list of estimators



voting_clf.estimators
# Updated list of trained estimators



voting_clf.estimators_
# We can either fit the VotingClassifier again, or just remove the SVM from the list of trained estimators:



del voting_clf.estimators_[2]
# Recheck



voting_clf.estimators_
# Evaluate the VotingClassifier again:



voting_clf.score(X_val, y_val)
# Set voting to "soft"



voting_clf.voting = "soft"

voting_clf.score(X_val, y_val)
# Test set



voting_clf.score(X_test, y_test)
[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
from sklearn.tree import DecisionTreeClassifier # Base Classifier for Bagging Method

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score



# Train an ensemble of 500 Decision Tree classifiers.

# BaggingClassifier automatically performs soft voting if the base classifier has a predict_proba() method.

# To train each predictor on a random subset of the input features, use max_features(0-1). Useful if training set has high dimentional features.

# "n_estimators = 500" was taking lot of CPU time, I used "n_estimators = 10" for demonstration purpose.

# bootstrap = True (Bagging)

clf_bagging = BaggingClassifier( DecisionTreeClassifier(), n_estimators = 10, max_samples = 0.8, bootstrap = True, oob_score=True, n_jobs = -1)

# bootstrap = False (Pasting)

clf_pasting = BaggingClassifier( DecisionTreeClassifier(), n_estimators = 10, max_samples = 0.8, bootstrap = False, n_jobs = -1)



clf_bagging.fit(X_train, y_train)

clf_pasting.fit(X_train, y_train)



y_pred_bagging = clf_bagging.predict(X_val)

y_pred_pasting = clf_bagging.predict(X_val)





clf_bagging.oob_score_, accuracy_score(y_val, y_pred_bagging), accuracy_score(y_val, y_pred_pasting)
# The decision function returns the class probabilities (if base estimator has a predict_proba() method) for each training instance. 

clf_bagging.oob_decision_function_[112]

#y_train[112]
# import matplotlib as mpl



# some_digit = X_train[112]

# some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")

# plt.axis("off")



# plt.show()
#  Trains a Random Forest classifier with 500 trees (each limited to maximum 16 nodes), using all available CPU cores:



from sklearn.ensemble import RandomForestClassifier



clf_randomforest = RandomForestClassifier(n_estimators=10, n_jobs=-1)

clf_randomforest.fit(X_train, y_train)

y_pred_rf = clf_randomforest.predict(X_val)



accuracy_score(y_val, y_pred_rf)
from sklearn.datasets import load_iris

iris = load_iris()

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):

    print(name, score)
# Train a classifier based on 200 Decision Trees 



from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)

ada_clf.fit(X_train, y_train)

y_pred_ada = ada_clf.predict(X_val)

accuracy_score(y_val, y_pred_ada)
import numpy as np



np.random.seed(42)

X = np.random.rand(100, 1) - 0.5

y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2)

tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)

tree_reg2 = DecisionTreeRegressor(max_depth=2)

tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)

tree_reg3 = DecisionTreeRegressor(max_depth=2)

tree_reg3.fit(X, y3)


X_new = np.array([[0.8]])



y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

y_pred
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

gbrt.fit(X, y)
gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)

gbrt_slow.fit(X, y)
import matplotlib.pyplot as plt





def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):

    x1 = np.linspace(axes[0], axes[1], 500)

    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)

    plt.plot(X[:, 0], y, data_style, label=data_label)

    plt.plot(x1, y_pred, style, linewidth=2, label=label)

    if label or data_label:

        plt.legend(loc="upper center", fontsize=16)

    plt.axis(axes)



plt.figure(figsize=(11,11))


plt.figure(figsize=(11,4))



plt.subplot(121)

plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")

plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)



plt.subplot(122)

plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])

plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)



#save_fig("gbrt_learning_rate_plot")

plt.show()
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)

gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)

gbrt_best.fit(X_train, y_train)
min_error = np.min(errors)
plt.figure(figsize=(11, 4))



plt.subplot(121)

plt.plot(errors, "b.-")

plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")

plt.plot([0, 120], [min_error, min_error], "k--")

plt.plot(bst_n_estimators, min_error, "ko")

plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)

plt.axis([0, 120, 0, 0.01])

plt.xlabel("Number of trees")

plt.title("Validation error", fontsize=14)



plt.subplot(122)

plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])

plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)



plt.show()
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)



min_val_error = float("inf")

error_going_up = 0

for n_estimators in range(1, 120):

    gbrt.n_estimators = n_estimators

    gbrt.fit(X_train, y_train)

    y_pred = gbrt.predict(X_val)

    val_error = mean_squared_error(y_val, y_pred)

    if val_error < min_val_error:

        min_val_error = val_error

        error_going_up = 0

    else:

        error_going_up += 1

        if error_going_up == 5:

            break  # early stopping
print(gbrt.n_estimators)
print("Minimum validation MSE:", min_val_error)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier
X_train_val, X_test, y_train_val, y_test = train_test_split( mnist.data, mnist.target, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split( X_train_val, y_train_val, test_size=10000, random_state=42)
random_forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

extra_trees_clf = ExtraTreesClassifier(n_estimators=10, random_state=42)

mlp_clf = MLPClassifier(random_state=42)
estimators = [random_forest_clf, extra_trees_clf, mlp_clf]

for estimator in estimators:

    print("Training the", estimator)

    estimator.fit(X_train, y_train)
[estimator.score(X_val, y_val) for estimator in estimators]
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)



for index, estimator in enumerate(estimators):

    X_val_predictions[:, index] = estimator.predict(X_val)
X_val_predictions
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)

rnd_forest_blender.fit(X_val_predictions, y_val)
rnd_forest_blender.oob_score_
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)



for index, estimator in enumerate(estimators):

    X_test_predictions[:, index] = estimator.predict(X_test)
y_pred = rnd_forest_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)