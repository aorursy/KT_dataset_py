import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# The following code creates and train a voting classifier in sklearn, composed of 3 divers classfier on the moon dataset
import sklearn
from sklearn.datasets import make_moons
x, y = make_moons()
x.shape, y.shape
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 12)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators = [('lr',log_clf), ('rf',rnd_clf),('svc',svm_clf)], voting = 'hard')

voting_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability = True)

voting_clf = VotingClassifier(estimators = [('lr',log_clf), ('rf',rnd_clf),('svc',svm_clf)], voting = 'soft')

voting_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))
# the following code trains an ensemble of 500 decision tree classifier, each
# training instances randomly sampled from the training set with replacement

# to use pasting set bootstrap =false


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500,
                          max_samples = 70, bootstrap = True, n_jobs = -1)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf, bag_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))
# Setting oob_score = True while creating a bagging classifier to request an 
# automatic oob evaluation after training.

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500,
                           bootstrap = True, n_jobs = -1, oob_score = True)
bag_clf.fit(x_train, y_train)
bag_clf.oob_score_
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred)
bag_clf.oob_decision_function_
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1)
rnd_clf.fit(x_train, y_train)

y_pred_rf = rnd_clf.predict(x_test)
# the following bagging classifier is roughly equivalent to the previous randomfores classifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter = 'random', max_leaf_nodes = 16),
                           n_estimators = 500, max_samples = 1.0, bootstrap = True ,n_jobs = -1)
bag_clf.fit(x_train, y_train)
rand_bag_clf = bag_clf.predict(x_test)
accuracy_score(y_test, y_pred)
from sklearn.ensemble import ExtraTreesClassifier

extr_cls = ExtraTreesClassifier()
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print (name, score)
# weighted error rate of the Jth predictor

def weighted(x):
    return x


def weighted_error_rate(data,target, predictor):
    no_of_samples = data.shape[0]
    predict_value = predictor.predict(data)
    weighted_value = [weighted(i) if i != j else int(1) for i,j in zip(predict_value, target)]
    normal_weight = [weighted(i) for i in predict_value]
    solution = [a/b for a,b in zip(weighted_value, normal_weight)]
    return solution
    
def predicators_weight(data,target,predictor,learning_param = 0.3):
    compute_1 = weighted_error_rate(data,target,predictor)
    compute_2 = np.log((1-compute_1)/compute_1)
    return learning_param * compute_2
def alpha(val):
    return val

def updated_weights(predicted_value, value):
    if predicted_value == value:
        val = weighted(predicted_value)
    else:
        val = weighted(predicted_value) * np.exp(alpha(value))
    return val
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 200, 
                            algorithm = 'SAMME.R', learning_rate = 0.5)
ada_clf.fit(x_train, y_train)
