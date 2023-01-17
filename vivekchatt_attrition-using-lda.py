# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data.head()
plt.figure(figsize=(12,7))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
plt.figure(figsize=(14,10))
sns.heatmap(data.corr(),yticklabels=False,cbar=True,linewidths=0)
plt.show()
plt.figure(figsize=(8,8))
sns.barplot(x=data['Department'],y=data['DailyRate'],hue=data['EducationField'])
plt.show()
sns.countplot(data['Attrition'])
plt.show()
data.columns
data.info()
BusinessTravel = pd.get_dummies(data['BusinessTravel'],drop_first=True)
Department = pd.get_dummies(data['Department'],drop_first=True)
EducationField = pd.get_dummies(data['EducationField'],drop_first=True)
Gender = pd.get_dummies(data['Gender'],drop_first=True)
JobRole  = pd.get_dummies(data['JobRole'],drop_first=True)
MaritalStatus = pd.get_dummies(data['MaritalStatus'],drop_first=True)
Train = data
def StrToBin(a):
    if a == 'Yes':
        return 1
    else:
        return 0
    
def StrToBinb(a):
    if a == 'Y':
        return 1
    else:
        return 0
    
Train['Attrition']=Train['Attrition'].apply(StrToBin)
Train['OverTime']=Train['OverTime'].apply(StrToBin)
Train['Over18']=Train['Over18'].apply(StrToBinb)
Train.info()
Train.drop(['Department','EducationField','Gender','BusinessTravel','JobRole','MaritalStatus'],axis=1,inplace=True)
Train = pd.concat([Train,Department,EducationField,Gender,BusinessTravel,JobRole,MaritalStatus],axis=1)
Train.info()
m = list(Train.columns)
n = list(filter(lambda t: t not in ['Attrition'], m))
X = Train[n]
y = Train['Attrition']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', solver='sag', C=1)
fit1 = classifier.fit(X_train, y_train)
y_pred = fit1.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2,solver='eigen',shrinkage='auto')
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=17,weights='distance',algorithm='brute')
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2,solver='svd')
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
from sklearn.svm import SVC
model = SVC(C=100,kernel = 'rbf')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier as DTC
dt = DTC(criterion='entropy',splitter='random')
tree = dt.fit(X_train,y_train)
pred_t = dt.predict(X_test)
cm = confusion_matrix(y_test,pred_t)
cr = classification_report(y_test,pred_t)
print(cm)
print(cr)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = fit1, X = X_train, y = y_train, cv = 10)
print('mean: ' ,accuracies.mean())
print('SD:' ,accuracies.std())
print(np.mean(accuracies))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
print('mean: ' ,accuracies.mean())
print('SD:' ,accuracies.std())
print(np.mean(accuracies))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print('mean: ' ,accuracies.mean())
print('SD:' ,accuracies.std())
print(np.mean(accuracies))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = tree, X = X_train, y = y_train, cv = 10)
print('mean: ' ,accuracies.mean())
print('SD:' ,accuracies.std())
print(np.mean(accuracies))