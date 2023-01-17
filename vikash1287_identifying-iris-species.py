

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


dataset= pd.read_csv("../input/Iris.csv")
dataset.head(10)
dataset.shape
#x : independent variable
#y : dependent variable
x= dataset.iloc[:,:-1].values
y=dataset.iloc[:,5].values
x
y
total=dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent],axis=1,keys=["Total","Percent"])
print(missing_data)
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y=label_y.fit_transform(y)
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)
# Fitting the classifier to the dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import  confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier2.fit(x_train,y_train)
y_pred2=classifier2.predict(x_test)
cm = confusion_matrix(y_test,y_pred2)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
# Random forest model seems to be perfect??
#Let's do k-fold cross validation to test it on different samples
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator=classifier2,X=x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()















