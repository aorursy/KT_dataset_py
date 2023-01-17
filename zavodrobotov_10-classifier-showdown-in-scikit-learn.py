import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



#import sklearn



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Swiss army knife function to organize the data



def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)  

    test = test.drop(['id'], axis=1)

    

    return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)

train.head(1)
sss = StratifiedShuffleSplit(labels, 1, test_size=0.5, train_size=0.5, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis







classifiers = [

    KNeighborsClassifier(1),

  #  KNeighborsClassifier(3),

   # KNeighborsClassifier(5),

   # KNeighborsClassifier(10),

    SVC(kernel="poly", degree = 1, probability=True, decision_function_shape = 'ovo', verbose = True),

    SVC(kernel="poly", degree = 1, probability=True, decision_function_shape = 'ovr', verbose = True),

    SVC(kernel="poly", degree = 1, C=0.25, probability=True),

    SVC(kernel="poly", degree = 1, C=5.25, probability=True, verbose = True),

    #SVC(kernel="poly", degree = 2, probability=True),

    #SVC(kernel="poly", degree = 10, probability=True),

    #SVC(kernel="linear", C=0.025, probability=True),

    SVC(kernel="rbf", C=0.025, probability=True,),

    #SVC(kernel="rbf", C=1.0, probability=True),

    #SVC(kernel="rbf", C=100.0, probability=True),

    #SVC(kernel="rbf", C=0.25, probability=True),

    #SVC(kernel="rbf", C=0.01, probability=True),

    

    #NuSVC(probability=True),

    #DecisionTreeClassifier(),

    #DecisionTreeClassifier(max_depth=30),

    #DecisionTreeClassifier(max_depth=33),

    #DecisionTreeClassifier(max_depth=35),

    #DecisionTreeClassifier(max_depth=37),

    #DecisionTreeClassifier(max_depth=40),

    #DecisionTreeClassifier(max_depth=45),

    #DecisionTreeClassifier(max_depth=60),

    #DecisionTreeClassifier(max_depth=80)

    #RandomForestClassifier(),

    #AdaBoostClassifier(),

    #GradientBoostingClassifier(),

    #GaussianNB(),

    LinearDiscriminantAnalysis(),

    #LinearDiscriminantAnalysis(),

    #LinearDiscriminantAnalysis(solver = 'eigen' ),

    #LinearDiscriminantAnalysis(solver = 'lsqr' ),

    #LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 0.00001),

    LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 0.000001),

    #LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 0.0000001),

    #LinearDiscriminantAnalysis(solver = 'eigen', shrinkage = 0.00000001),

    #LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 0.000001),

    #LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 0.00000001)

    #QuadraticDiscriminantAnalysis()

]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



print(sklearn.__version__)



#conda list scikit-learn



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
# Predict Test Set

favorite_clf = LinearDiscriminantAnalysis()

favorite_clf.fit(X_train, y_train)

test_predictions = favorite_clf.predict_proba(test)



# Format DataFrame

submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()



# Export Submission

#submission.to_csv('submission.csv', index = False)

submission.tail()