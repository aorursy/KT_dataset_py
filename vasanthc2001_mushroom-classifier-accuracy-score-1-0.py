import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data.sample(10)
data.shape
data.isna().any()
y=data['class']

X=data.drop(['class'],axis=1)
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
label_encoder=LabelEncoder()

X=X.apply(lambda x:label_encoder.fit_transform(x))

y=label_encoder.fit_transform(y)
X.head()
y
from sklearn.feature_selection import RFE

from sklearn.linear_model import SGDClassifier

classifier=SGDClassifier(alpha=0.01)

rfe=RFE(classifier,step=1,n_features_to_select=10)

rfe.fit(X,y)
rfe.ranking_
X=X.iloc[:,[3,5,6,7,9,10,11,12,16,17,19,20,21]]
X
one_hot_encoder=OneHotEncoder(sparse=True,drop='first')

X=one_hot_encoder.fit_transform(X)
X=X.toarray()
X.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=10,min_samples_leaf=100)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
from sklearn.svm import SVC

clf1=SVC(C=0.1)

clf1.fit(X_train,y_train)
y_pred=clf1.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
accuracy_score(y_test,y_pred)
from sklearn.linear_model import SGDClassifier

clf2=SGDClassifier(alpha=0.01)

clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
accuracy_score(y_test,y_pred)
from sklearn.neighbors import KNeighborsClassifier

clf3=KNeighborsClassifier(n_neighbors=5)

clf3.fit(X_train,y_train)
y_pred=clf3.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
accuracy_score(y_test,y_pred)