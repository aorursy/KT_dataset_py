import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pylab as pl

from sklearn.preprocessing import LabelEncoder 
df_train = pd.read_csv('../input/titanic/train.csv')

df_train.info()
print(df_train.columns)

df_train = df_train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

df_train.dropna()

df_train["Sex"] = LabelEncoder().fit_transform(df_train["Sex"].astype('str'))

df_train["Embarked"] = LabelEncoder().fit_transform(df_train["Embarked"].astype('str'))
#checking correlation between selected features

sns.heatmap(df_train.corr(), annot=True) 
y = df_train["Survived"]

X = df_train.drop(["Survived"],axis=1)

X.fillna(X.mean(),inplace =True)

print(X.isnull().any())
print(y.isnull().any())

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=5)

# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=5)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,roc_curve,confusion_matrix,auc

target_names = ["ALIVE","DEAD"]
ra_dict = {}

def generate_metrics(clf_name,y_true,y_pred,target_names):

  print("Classification Report : \n{}".format(classification_report(y_test, y_pred, target_names=target_names)))

  print("Confusion Matrix : \n{}".format(confusion_matrix(y_test,y_pred)))

  fpr, tpr, thresholds = roc_curve(y_true,y_pred)

  roc_auc  = auc(fpr, tpr)

  ra_dict[clf_name] = roc_auc

  label =  clf_name +' (area = '+ str(roc_auc)
gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

generate_metrics("GaussianNB",y_test,y_pred,target_names)
lr = LogisticRegression(penalty='l2', tol=0.01, random_state = 3) 

lr.fit(X_train,y_train) 

y_pred = lr.predict(X_test)

generate_metrics("Logistic Regression",y_test,y_pred,target_names)
knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

generate_metrics("KNN",y_test,y_pred,target_names)
svc =SVC(kernel='rbf',gamma='auto')

clf_svc = make_pipeline(StandardScaler(), svc).fit(X_train,y_train)

y_pred = clf_svc.predict(X_test)

generate_metrics("SVM",y_test,y_pred,target_names)
dtf = DecisionTreeClassifier(random_state=3)

dtf.fit(X_train,y_train)

y_pred = dtf.predict(X_test)

generate_metrics("DecisionTreeClassifier",y_test,y_pred,target_names)
rfc = RandomForestClassifier(n_estimators = 85, max_features='auto', criterion='entropy',max_depth=4)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

generate_metrics("RandomForestClassifier",y_test,y_pred,target_names)
from pprint import pprint

pprint(ra_dict)

clf_max = max(ra_dict, key=ra_dict.get) 

print(clf_max)
df_test = pd.read_csv('../input/titanic/test.csv')

print(df_test.shape)

submission = pd.DataFrame(df_test['PassengerId'])

df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)

df_test["Sex"] = LabelEncoder().fit_transform(df_test["Sex"].astype('str'))

df_test["Embarked"] = LabelEncoder().fit_transform(df_test["Embarked"].astype('str'))

df_test.fillna(df_test.mean(),inplace=True)
print(df_test.isnull().any())

result = rfc.predict(df_test)

submission['Survived'] = result
submission.to_csv('submission.csv', encoding='utf-8', index=False)

submission.tail()