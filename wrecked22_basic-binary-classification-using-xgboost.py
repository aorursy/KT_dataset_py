import numpy as np 

import pandas as pd 

import os

import tensorflow as tf

import pandas as pd

import itertools

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix # Loading required libraries
data = pd.read_csv("../input/diabetes.csv") # Load the dataset
data.head(5) # Quick look 
sns.countplot(data.Outcome)

plt.title('Categories') # Class imbalance is there but not considerable
y = data.Outcome

X = data.drop("Outcome", axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size=0.3, 

                                                    random_state=42) # Spliting Train Test datasets
clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=90,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test) #Let's try to fit the Random Forest model on this Data
metrics.accuracy_score(y_test, y_pred) #  Not bad 
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = clf.predict(X_test)

confusion_mtx = confusion_matrix(y_test, y_pred) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 

# Confusion Matrix 
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=7,

              min_child_weight=1, missing=None, n_estimators=7, n_jobs=-1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=2)

model.fit(X_train, y_train)

#Let's try to fit the XGBosst classifier model on this Data
y_pred = model.predict(X_test)

metrics.accuracy_score(y_test, y_pred) # Better than Random Forest
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



y_pred = model.predict(X_test)

#y_pred_classes = np.argmax(y_pred,axis = 1) 

confusion_mtx = confusion_matrix(y_test, y_pred) 

plot_confusion_matrix(confusion_mtx, classes = range(2)) 

# Again Plotting Confusion Matrix