# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
Train = pd.read_csv("/kaggle/input/titanic/train.csv")
Test = pd.read_csv("/kaggle/input/titanic/test.csv")
Train.describe()
Train.info()
Train
sns.barplot(x=Train["Pclass"],y=Train["Survived"])
sns.barplot(x=Train["Sex"],y=Train["Survived"])
sns.barplot(x=Train["SibSp"],y=Train["Survived"])
sns.barplot(x=Train["Parch"],y=Train["Survived"])
Train["Parch"].unique()
sns.distplot(a=Train[Train["Survived"]==1]["Age"])
sns.distplot(a=Train[Train["Survived"]==0]["Age"])

plt.figure(figsize=(10,10))
sns.scatterplot(x=Train["Fare"],y=Train["Age"],hue=Train["Survived"])
sns.barplot(x=Train["Embarked"],y=Train["Survived"])
#---Features Selection---
X_train = Train.iloc[:,[2,4,5,9,11,6]].values
y_train = Train.iloc[:,1].values
test  = Test.iloc[:,[1,3,4,8,10,5]].values
#Handle Missing value
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = SimpleImputer(strategy='most_frequent')

#---Handle Missing Value in Train----
imputer = imputer.fit(X_train[:,2:4])
imp = imp.fit(X_train[:,[4]])
X_train[:,2:4] = imputer.transform(X_train[:,2:4])
X_train[:,[4]] = imp.transform(X_train[:,[4]])

#---Handle Missing Value in Test
imputer = imputer.fit(test[:,2:4])
test[:,2:4] = imputer.transform(test[:,2:4])


#Creating Dummy Variable for Categorical Variables
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1,4])], remainder='passthrough')
X_train=np.array(columnTransformer.fit_transform(X_train))
test = np.array(columnTransformer.transform(test))

#Delete one data from dummies to remove dependencies
X_train = X_train[:,[1,2,3,5,6,7,8]]
test = test[:,[1,2,3,5,6,7,8]]


#Features Scalling to remove domination of any one perticuler feature
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
test = sc_X.transform(test)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=24)
#Simple Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("Logistic.csv", index=False)
#---Naive Bais Classification---
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("Naive.csv", index=False)
#---SVM Classification---
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=178)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("SVM.csv", index=False)
#---Kernel SVM Classification---
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("KSVM.csv", index=False)
#---KNN Classification---
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("KNN.csv", index=False)
#---Decision Tree Classification---
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("DecisionTree.csv", index=False)
#---Random Forest Classification---
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000,criterion = "entropy", random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("RandomForest.csv", index=False)
#---PCA---
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=3,kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

#---Kernel SVM Classification---
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
yt_pred = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)

y_res = classifier.predict(kpca.transform(test))
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("PCA with KSVM.csv", index=False)
#---XGBOOST Classification----
from xgboost.sklearn import XGBClassifier
model = XGBClassifier(learning_rate=0.001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
yt_pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print("Test Acc : ",acc)
print(cm)
cm = confusion_matrix(y_train,yt_pred)
acc = metrics.accuracy_score(y_train,yt_pred)
print("Train Acc : ",acc)
print(cm)


y_res = model.predict(test)
df = pd.DataFrame({'PassengerId':Test.iloc[:,0].values,'Survived':y_res})
df.to_csv("XBoost.csv", index=False)

