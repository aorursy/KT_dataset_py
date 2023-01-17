#importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# We are reading our data
data = pd.read_csv("../input/heart.csv")
#splitting the given data into feautures and target columns
X,y = data.iloc[:,:-1],data.iloc[:,[-1]]
#importing traintest split and splitting given data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)
#importing logistic regression from sklearn.linearmodel
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#training the given model using training data
lr.fit(X_train,y_train)
#prdicting y values for X_test
y_pred = lr.predict(X_test)
#importing classification report, confusion matrix and accuracy score 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
#importing RandomForestClassifier form sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
#training the classifier using training data
rfc.fit(X_train,y_train)
#predicting y values for x_test values
y_pred = rfc.predict(X_test)
#importing classification report, confusion matrix and accuracy score 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
# importing DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier  
dtc = DecisionTreeClassifier(criterion='gini') 
#Training the classifier with X_train and y_train
dtc.fit(X_train, y_train)
#predicting the values of y with respect to X_test 

y_pred = dtc.predict(X_test)
#importing classification report, confusion matrix and accuracy score 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
#importing SVC from sklearn.svm
from sklearn.svm import SVC
svc = SVC()
#training the classifier using training data
svc.fit(X_train,y_train)
#predicting the values of y for x_test
y_pred = svc.predict(X_test)
#importing classification report, confusion matrix and accuracy score 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
#importing KNeighboursClassifier from sklearn.neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
#training the classifier using training data
knn.fit(X_train,y_train)
#predicting y values for x_test
y_pred = knn.predict(X_test)
#importing classification report, confusion matrix and accuracy score 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))