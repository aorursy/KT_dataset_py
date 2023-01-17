# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loadind the dataset
train = pd.read_csv("/kaggle/input/train.csv")
##print first 5 records of dataset
train.head()
#find the number of rows and columns of data
train.shape
print("No of passenger in original data:"+str(len(train)))
#find the total missing values of individuals of columns of dataset
train.isnull().sum()
#drop unwanted column
train.drop("Cabin",axis=1,inplace=True)
# fill missing values with mean column values
train['Age'].fillna((train["Age"].mean()),inplace=True)
#turns type columns intoma dummy variable
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
pcl=pd.get_dummies(train['Pclass'],drop_first=True)
train=pd.concat([train,sex,pcl,embark],axis=1)
train.head()
#Dropping the categorical variables
train.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
train.head()
#features selection
#feature slection
X=train.drop('Survived',axis=True)#independent columns
y=train['Survived'] #target column
#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
#Standard scaling of data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#Applying RandonForest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
model = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Confusion Matrix Evaluation Metrics
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(y_test, y_pred))
print("Precision=",metrics.precision_score(y_test, y_pred))
print("Recall=",metrics.recall_score(y_test, y_pred))
#Applying LogisticRegression 
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state = 0)
lr_clf.fit(X_train, y_train)
# Predicting the Test set results
y_pred = lr_clf.predict(X_test)
#Confusion Matrix Evaluation Metrics
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(y_test, y_pred))
print("Precision=",metrics.precision_score(y_test, y_pred))
print("Recall=",metrics.recall_score(y_test, y_pred))
#Applying Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf= GaussianNB()
nb_clf.fit(X_train, y_train)
# Predicting the Test set results
y_pred = lr_clf.predict(X_test)
#Confusion Matrix Evaluation Metrics
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(y_test, y_pred))
print("Precision=",metrics.precision_score(y_test, y_pred))
print("Recall=",metrics.recall_score(y_test, y_pred))
