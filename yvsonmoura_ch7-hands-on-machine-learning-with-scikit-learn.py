## Page 191/192

## Ensemble Learning
# if you aggregate the predictions of a group of predictors (such as classifiers or regressors), you will often get better predictions than with the best individual predictor
# A group of predictors is called an ENSEMBLE

## Random Forests
# For example, you can train a group of Decision Tree classifiers, each on a different
# random subset of the training set. To make predictions, you just obtain the predictions
# of all individual trees, then predict the class that gets the most votes. Such an ensemble of Decision Trees is called a Random Forest,

## Voting Classifiers
# A very simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes.
# This majority-vote classifier is called a HARD VOTING CLASSIFIER.

## Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms
# Weak Learner: It means that the classifier performs slightly better than random guessing
# Strong Learner: It means that the classifier achieves high accuracy
## Page 194

# Voting Classifiers - Hard Voting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# Data
moons = make_moons(n_samples=800, shuffle=True, noise=0.55, random_state=999)
X = moons[0]
y = moons[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred)) 

# The voting classifier slightly outperforms all the individual classifiers.
## Page 194

# Voting Classifiers - Soft Voting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import random


# Data
moons = make_moons(n_samples=800, shuffle=True, noise=0.55, random_state=999)
X = moons[0]
y = moons[1]

random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True) # probability=True enables 5-fold cross-validation and predict_proba may be inconsistent with predict.

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred)) 

# The soft voting method outperforms the hard voting method by increasing accuracy.
## Pages 195/196/197

## Bagging and Pasting with Scikit-Learn
# Another approach is to use the same training algorithm for every predictor, but to train them on different random subsets of the training set.
# When sampling is performed WITH REPLACEMENT, this method is called BAGGING (bootstrap aggregating).
# When sampling is performed WITHOUT REPLACEMENT, this method is called PASTING.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Data
moons = make_moons(n_samples=600, shuffle=True, noise=0.2, random_state=999)
X = moons[0]
y = moons[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_firstmoon = X[np.where(y == 0)] # First Moon
X_secondmoon = X[np.where(y == 1)] # Second Moon

# Meshgrid
XX = np.arange(-3, 3, 0.025)
X_bg = np.array([[0,0]])

for i in range(len(XX)):
    for j in range(len(XX)):
        X_bg = np.append(X_bg, [[XX[i], XX[j]]], axis=0)

# Decision Tree Classifier with Bagging
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_bg)

# Plot
plt.style.use('ggplot')
fig = plt.subplots(1, 1, figsize=(14, 7))

plt.scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='orange', marker='o', s=25, alpha=0.8)
plt.scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='blue', marker='s', s=25, alpha=0.8)
plt.xlabel(r'$X_1$', fontsize=20, color='k')
plt.ylabel(r'$X_2$', fontsize=20, color='k')
plt.xlim(-1.5, 2.5)
plt.ylim(-1, 1.5)
plt.title('Decision Trees with Bagging', fontsize=25, color='k')

# Background color - Predicted values
plt.scatter(X_bg[np.where(y_pred==0),0], X_bg[np.where(y_pred==0),1], color='yellow', marker='.', s=50, alpha=0.2)
plt.scatter(X_bg[np.where(y_pred==1),0], X_bg[np.where(y_pred==1),1], color='green', marker='.', s=50, alpha=0.2)
## Pages 197/198

# Out-of-Bag Evaluation
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train, y_train)

print('Bagging Classifier Accuracy for Out-of-Bag instances: ',bag_clf.oob_score_)

from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(X_test)
print('Bagging Classifier Accuracy for testing set: ',accuracy_score(y_test, y_pred))

#print('Out-of-bag probabilities for each instance: \n', bag_clf.oob_decision_function_) # Out-of-bag probabilities for each instance
## Page 198

## Random Patches and Random Subspaces - The BaggingClassifier class supports sampling the features as well (controlled by max_features and bootstrap_features hyperparameters)
# INSTANCE SAMPLING <> FEATURE SAMPLING
# Sampling both training instances and features is called the RANDOM PATCHES METHOD.
# Keeping all TRAINING INSTANCES (i.e., bootstrap=False and max_samples=1.0) but SAMPLING FEATURES (i.e., bootstrap_features=True and/or max_features smaller than 1.0) is called the RANDOM SUBSPACES METHOD.
# Sampling features results in even more predictor diversity, trading a bit more bias for a lower variance.

## Page 199

## Random Forests
# Instead of building a BaggingClassifier and passing it a DecisionTreeClassifier, you can instead use the RandomForestClassifier class, which is more convenient and optimized for Decision Trees
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
print('Random Forest Classifier Accuracy: ', accuracy_score(y_test, y_pred_rf))

bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
print('Bagging Classifier Accuracy: ', accuracy_score(y_test, y_pred_bag))

# Both classifiers have similar results.
## Pages 200/201

# Feature Importance 

from sklearn.datasets import load_iris

#Data
iris = load_iris()

# Model
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
## Page 201

# Example of Feature Importance with MNIST Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Data
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape #(70000, 784)
y.shape #(70000, )

# Model
rnd_clf_mnist = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf_mnist.fit(X, y)

# Plot
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

importance_digits = rnd_clf_mnist.feature_importances_.reshape(28,28)

first = axes[0].imshow(importance_digits, cmap = mpl.cm.binary, interpolation="none")
axes[0].axis("off")

second = axes[1].imshow(importance_digits, cmap = mpl.cm.afmhot, interpolation="none")
axes[1].axis("off")

# Colorbar
cbar = fig.colorbar(second, ax=axes[1], label='Feature Importance', shrink=0.73, ticks=[importance_digits.min(), importance_digits.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])  # vertically oriented colorbar

## Pages 201-205

## Boosting
# Boosting (originally called hypothesis boosting) refers to any Ensemble method that can combine several weak learners into a strong learner.
# The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.
# Most popular types of boosting methods are AdaBoost (short for Adaptive Boosting) and Gradient Boosting.

# AdaBoost (One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)

y_pred = [p for p in ada_clf.staged_predict(X_test)]

from sklearn.metrics import accuracy_score

acc_ada = []
for i, j in enumerate(y_pred):
    acc_ada.append(accuracy_score(y_test, y_pred[i]))

# Plot
fig, axes = plt.subplots(1, 1, figsize=(8, 4))

axes.plot(np.arange(1, 201), acc_ada, lw=1)
axes.set_xlim(0, 200)
axes.set_xticks([1,20,40,60,80,100,120,140,160,180,200])
axes.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
axes.set_ylabel('Accuracy', fontsize=12)
axes.set_title('AdaBoost Accuracy', fontsize=15)

## Pages 205/206

## Gradient Boosting - How it works
# This method tries to fit the new predictor to the residual errors made by the previous predictor.
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X_train, y_train)

y2 = y_train - tree_reg1.predict(X_train)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X_train, y2)

y3 = y2 - tree_reg2.predict(X_train)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X_train, y3)

# It can make predictions on a new instance simply by ADDING UP the PREDICTIONS of ALL the TREES
y_pred_step = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))

# Scikit-Learn GRBT Class for 
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X_train, y_train)

y_pred_GRBT = gbrt.predict(X_test)

(y_pred_step.round(4) == y_pred_GRBT.round(4)).all() # Verifying compatibility between arrays


## Pages 208/209

## Finding the optimal number of trees
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data
moons = make_moons(n_samples=400, shuffle=False, noise=0.07)
X = moons[0]
y = moons[1]
X1 = moons[0][np.where(moons[1]==1), 0].reshape(200, 1)
X2 = moons[0][np.where(moons[1]==1), 1].reshape(200,)
y1 = moons[1][np.where(moons[1]==1)]

X_train, X_val, y_train, y_val = train_test_split(X1, X2)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(np.arange(1, 121), errors, lw=1.5, c='b')
axes[0].hlines(y=min(errors), xmin=0, xmax=120, ls='--', lw=1)
axes[0].set_xlim(0, 120)
axes[0].set_xticks(np.arange(0, 140, 20))
axes[0].set_xlabel('Number of Trees (n_estimators)', fontsize=12)
axes[0].set_ylabel('Error (MSE)', fontsize=12)
axes[0].set_title('Validation Error', fontsize=15)
axes[0].text(1, min(errors)+0.0015, "Minimum Error", family="serif", fontsize=7)

XA = np.arange(-0.5, 2.5, 0.01)
XA = XA.reshape(len(XA), 1)

axes[1].scatter(X1, X2, c='b', s=5)
axes[1].plot(XA, gbrt_best.predict(XA), c='r')
axes[1].set_xlabel(r'$X_1$', fontsize=12)
axes[1].set_ylabel(r'$y$', fontsize=12)
axes[1].set_title('Best Model ({0} estimators)' .format(bst_n_estimators), fontsize=15)
## Page 209

## Early Stop
# The following code stops training when the validation error does not improve for five iterations in a row

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
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
            break # early stopping
        
print('Minimum Error Value:', min_val_error)

## Stochastic Gradient Boosting
# The GradientBoostingRegressor class also supports a SUBSAMPLE hyperparameter, which specifies the fraction of training instances to be used for training each tree.
# if subsample=0.25, then each tree is trained on 25% of the training instances, selected randomly.
## Page 210

## Extreme Gradient Boosting (XGBoost)
# This package was initially developed by Tianqi Chen as part of the Distributed (Deep) Machine Learning Community (DMLC), and it aims at being extremely fast, scalable and portable.
import xgboost

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)

# Early Stopping feature
xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)

## Pages 210/211/212

## Stacking - stacked generalization
# Instead of using trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble, why don’t we train a model to perform this aggregation?

# To train the blender (final predictor), a common approach is to use a hold-out set. First, the training set is split in two subsets.
# The first subset is used to train the predictors in the first layer. Next, the first layer predictors are used to make predictions on the second (held-out) set.
# Now for each instance in the hold-out set there are three predicted values.
# We can create a new training set using these predicted values as input features (which makes this new training set three-dimensional), and keeping the target values.
# The blender is trained on this new training set, so it learns to predict the target value given the first layer’s predictions.