# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Iris.csv')
df.drop(columns='Id',inplace=True)
df.head()
df.shape
df.info()
df.describe()
df.describe(include=['object'])
df['Species'].value_counts()
correlation = df.iloc[:,1:-1].corr()
correlation
df.skew()
df.isnull().sum()
fig, ax = plt.subplots(2,2, sharex=True, figsize=(12, 4))
df.hist(figsize=(8,4),ax=ax)
plt.show()
_, ax = plt.subplots(2,2, sharex=True, figsize=(12, 4))
df.plot(kind='Density',ax=ax,subplots=True)
plt.show()
_, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4))
sns.distplot(df['SepalLengthCm'],ax=ax[0])
sns.boxplot(df['SepalLengthCm'],ax=ax[1])
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.show()
sns.pairplot(df,hue='Species')
plt.show()
X = df.drop('Species',axis=1)
y = df['Species'].as_matrix()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y_label = label.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_label,test_size=30/150,random_state=3)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score,confusion_matrix
LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)
y_pred = LogReg.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.show()
svc = LinearSVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.show()
svc = SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.show()