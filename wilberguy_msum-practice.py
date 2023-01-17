# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as skl # machine learning
from sklearn import preprocessing # preprocessing
import tensorflow as tf # neural networks

import seaborn as sns # visuals and display
import matplotlib.pyplot as plt # display

import plotly.graph_objs as go # D3 display
from plotly.offline import * # offline
init_notebook_mode(connected=True) # for Notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/mudac-2018-prelims/train.csv')
df.head()
df.info()
# create independent and dependent variables
y_train = df['Diabetes']
X_train = df.iloc[:,:-1] # all columns but Diabetes
X_test = df = pd.read_csv('../input/localtest/test.csv')
x.describe()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Build Models -----------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# ----------- Classifiers ------------

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rfc.fit(X_train, y_train)

# Decision Tree Classifier
# --- DOES NOT USE ECULDIAN DISTANCES
# THEREFORE SCALED FEATURES ARE NOT REQUIRED
# TO PLOT IT WE LEAVE IS SCALED -------
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train, y_train)
#Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
# Supoort Vector Classifier Kernel
svck = SVC(kernel = 'rbf' )
svck.fit(X_train, y_train)
# Support Vector Classifier
svc = SVC(kernel = 'linear')
svc.fit(X_train, y_train)
# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 17, metric = 'minkowski', p = 4)
knn.fit(X_train, y_train)


# Predicting the Test set results
knn_y_pred = knn.predict(X_test)
svc_y_pred = svc.predict(X_test)
svck_y_pred = svck.predict(X_test)
nb_y_pred = nb.predict(X_test)
dtc_y_pred = dtc.predict(X_test)

predictions = (knn_y_pred, svc_y_pred, svck_y_pred, nb_y_pred, dtc_y_pred)
models = (knn,svc, svck, nb, dtc)
for classifier in models:
    
    # Cross Validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print('{}:'.format(classifier), scores.mean(), '\n''-------------','\n')
        # Test Accuaracy of our Models
    print(str(classifier),': ', classifier.score(X_test, y_test))
    # OR using just the data
for pred in predictions:
    
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, pred))
    # OR the most insightful report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, pred))
    
    
    # ROC AUC Scores
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(y_test, pred))
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pred)
    
pred = pd.DataFrame(nb_y_pred, columns = ['Prediction'], dtype= 'int' )
pred['ID'] = range(511, len(pred) + 511)
pred.head(-1)
pred.to_csv('NaivesBayes.csv', encoding='utf-8', index=False)
