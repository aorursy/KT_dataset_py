

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline






wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine.head(20)
wine.tail(20)
wine.info()
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'pH', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
wine.quality = [1 if each >= 7 else 0 for each in wine.quality]
wine.quality
wine.quality.value_counts()
sns.countplot(wine.quality)
y = wine.quality.values
y
x_data = wine.drop(["quality"],axis=1)
x_data
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
RF = RandomForestClassifier(n_estimators=200, random_state=1)
RF.fit(x_train,y_train)
predictions = RF.predict(x_test)
score = round(accuracy_score(y_test, predictions), 5)
print("Random Forest Score {}".format(score))

from sklearn.model_selection import  cross_val_score
RF_CrossValidation = cross_val_score(estimator=RF, X=x, y=y, cv = 40)
print("Random Forest Cross Validation Score ",RF_CrossValidation.max())
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier(penalty=None)
SGD.fit(x_train,y_train)
predictions = SGD.predict(x_test)
score = round(accuracy_score(y_test,predictions),5)
print("Stochastic Gradient Decent Score {}".format(score))
from sklearn.svm import SVC
SVM = SVC(random_state=1)
SVM.fit(x_train,y_train)
predictions = SVM.predict(x_test)
score = round(accuracy_score(y_test,predictions),5)
print("Support Vector Machine Score {}".format(score))
from sklearn.model_selection import GridSearchCV
param = {
    'C'     :[0.1,0.5,0.9,1,1.5,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
SVM_GridSearchCV = GridSearchCV(SVM, param_grid=param, scoring='accuracy', cv=40)
SVM_GridSearchCV.fit(x_train,y_train)
SVM_GridSearchCV.best_params_
SVM_BestGridSearchCV1 = SVC(C = 1.5, gamma = 1.3, kernel = 'rbf')
SVM_BestGridSearchCV1.fit(x_train,y_train)
predictions = SVM_BestGridSearchCV1.predict(x_test)
score = round(accuracy_score(y_test,predictions),5)
print("Support Vector Machine Grid Search CV Score {}".format(score))