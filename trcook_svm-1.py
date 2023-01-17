import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import time
from operator import itemgetter
import os
from sklearn import preprocessing
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


#import data
df = pd.read_csv("../input/week-2-data/Churn_Calls.csv", sep=',')
df.head(10)
# See each collum name
print(df.columns)
df.shape
# designate target variable name
targetName = 'churn'
# move target variable into first column
targetSeries = df[targetName]
del df[targetName]
df.insert(0, targetName, targetSeries)
expected=targetName
df.head(10)
gb = df.groupby(targetName)
targetEDA=gb[targetName].aggregate(len)
plt.figure()
targetEDA.plot(kind='bar', grid=False)
plt.axhline(0, color='k')

le_dep = preprocessing.LabelEncoder()
#to convert into numbers
df['churn'] = le_dep.fit_transform(df['churn'])
# perform data transformation
for col in df.columns[1:]:
	attName = col
	dType = df[col].dtype
	missing = pd.isnull(df[col]).any()
	uniqueCount = len(df[attName].value_counts(normalize=False))
	# discretize (create dummies)
	if dType == object:
		df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
		del df[attName]
# split dataset into testing and training
features_train, features_test, target_train, target_test = train_test_split(
    df.iloc[:,1:].values, df.iloc[:,0].values, test_size=0.40, random_state=0)
print(features_test.shape)
print(features_train.shape)
print(target_test.shape)
print(target_train.shape)
print("Percent of Target that is Yes", target_test.mean())
#data.groupby(['col1', 'col2'])

clf_linSVC=LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=1.0, class_weight='balanced')
clf_linSVC.fit(features_train, target_train)
predicted_SVC=clf_linSVC.predict(features_test)
expected = target_test
# summarize the fit of the model
print(classification_report(expected, predicted_SVC,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVC))
print(accuracy_score(expected,predicted_SVC))
start_time = time.clock()

#standard linear SVC
clf_lin = SVC(kernel='linear', C=1.0,class_weight=None)
clf_lin.fit(features_train, target_train)
predicted_SVM=clf_lin.predict(features_test)
expected = target_test
# summarize the fit of the model
print(classification_report(expected, predicted_SVM,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVM))
print(accuracy_score(expected,predicted_SVM))
print(time.clock() - start_time, "seconds")
start_time = time.clock()

#standard linear SVC
clf_lin = SVC(kernel='linear', C=1.0,class_weight='balanced')
clf_lin.fit(features_train, target_train)
predicted_SVM=clf_lin.predict(features_test)
expected = target_test
# summarize the fit of the model
print(classification_report(expected, predicted_SVM,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVM))
print(accuracy_score(expected,predicted_SVM))
print("Time to run", time.clock() - start_time, "seconds")
start_time = time.clock()

parameters = {'C':[.01,.05,1,3,4,9,10]}
svr = SVC(kernel='linear')
grid_svm = GridSearchCV(svr, parameters, cv=5)
grid_svm.fit(features_train, target_train)
print("SCORES", grid_svm.cv_results_)
print("BEST SCORE", grid_svm.best_score_)
print("BEST PARAM", grid_svm.best_params_)
print("Time to run", time.clock() - start_time, "seconds")
start_time = time.clock()

parameters = {'kernel':('linear', 'rbf'), 'C':[.001,.01,1,3,5,10]}
svr = SVC()
grid_svm = GridSearchCV(svr, parameters, cv=5)
grid_svm.fit(features_train, target_train)
print("SCORES", grid_svm.cv_results_)
print("BEST Estm",grid_svm.best_estimator_) 
print("BEST SCORE",grid_svm.best_score_)
print("BEST PARAM", grid_svm.best_params_)
print("Time to run", time.clock() - start_time, "seconds")
start_time = time.clock()

#standard linear SVC
clf_lin = SVC(kernel='linear', C=10.0,class_weight='balanced',gamma='auto')
clf_lin.fit(features_train, target_train)
predicted_SVM=clf_lin.predict(features_test)
expected = target_test
# summarize the fit of the model
print(classification_report(expected, predicted_SVM,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_SVM))
print(accuracy_score(expected,predicted_SVM))
print("Time to run", time.clock() - start_time, "seconds")
start_time = time.clock()

#standard linear SVC
clf_rbf = SVC(kernel='rbf', C=1.0, degree=3, class_weight='balanced',gamma=0.1)
clf_rbf.fit(features_train, target_train)
predicted_rbf=clf_rbf.predict(features_test)
expected = target_test
# summarize the fit of the model
print(classification_report(expected, predicted_rbf,target_names=['No', 'Yes']))
print(confusion_matrix(expected, predicted_rbf))
print(accuracy_score(expected,predicted_rbf))
print("Time to run", time.clock() - start_time, "seconds")


clf_poly = SVC(kernel='poly', degree=2, C=1.0,class_weight='balanced')
clf_poly.fit(features_train, target_train)
predicted_poly=clf_poly.predict(features_test)
expected = target_test
#summarize the fit of the model
print(classification_report(expected, predicted_poly,target_names=['No', 'Yes']))
# print(confusion_matrix(expected, predicted_poly))
print(accuracy_score(expected,predicted_poly))