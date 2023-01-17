# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from IPython.display import display, Image
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
df.info()
df.describe()
sns.pairplot(df)
sns.countplot(x='quality',data=df)
sns.boxplot('quality','fixed acidity',data=df)
sns.boxplot('quality', 'volatile acidity', data = df)
sns.boxplot('quality', 'citric acid', data = df)
sns.boxplot('quality', 'residual sugar', data = df)
sns.boxplot('quality', 'chlorides', data = df)
sns.boxplot('quality', 'free sulfur dioxide', data = df)
sns.boxplot('quality', 'total sulfur dioxide', data = df)
sns.boxplot('quality', 'density', data = df)
sns.boxplot('quality', 'pH', data = df)
sns.boxplot('quality', 'sulphates', data = df)
sns.boxplot('quality', 'alcohol', data = df)
#New column is created as NewQuality. It has the values 1,2, and 3. 
#1 - Bad quality
#2 - Average quality
#3 - Excellent quality
#Split the dataset 
#1,2,3 - Bad quality
#4,5,6,7 - Average quality
#8,9,10 - Excellent quality

#Create an empty list called NewQuality
NewQuality = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        NewQuality.append('1')
    elif i >= 4 and i <= 7:
        NewQuality.append('2')
    elif i >= 8 and i <= 10:
        NewQuality.append('3')
df['NewQuality'] = NewQuality
x = df.iloc[:,:11]
y = df['NewQuality']
x.head()
y.head()
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)
#print confusion matrix and accuracy score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score*100)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_predict=nb.predict(X_test)
#print confusion matrix and accuracy score
nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_predict=rf.predict(X_test)
#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)
from sklearn.svm import SVC
#we shall use the rbf kernel first and check the accuracy
lin_svc = SVC()
lin_svc.fit(X_train, y_train)
lin_svc=rf.predict(X_test)
#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
lin_svc_acc_score = accuracy_score(y_test, rf_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)