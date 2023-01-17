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
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.head()
df.info
df.describe()
df.shape
#checking for null values
df.isna().apply(pd.value_counts)
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
sns.distplot(df['sepal_length'])
sns.distplot(df['sepal_width'])
sns.distplot(df['petal_length'])
sns.distplot(df['petal_width'])
df.skew()
sns.pairplot(df, hue='species')
sns.boxplot(df['sepal_length'])
sns.boxplot(df['sepal_width'])
sns.boxplot(df['petal_length'])
sns.boxplot(df['petal_width'])
corr = df.corr()
corr
plt.figure(figsize = (20,10))
sns.heatmap(corr, cmap='RdYlGn', vmax = 1.0, vmin = -1.0)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
X = df.iloc[0:,:4]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape)
print(X_test.shape)
# import logistic regression and training data set
model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
accuracy_score(y_test,y_pred)
#Classification Report
print(classification_report(y_test, y_pred, digits = 2))
from sklearn.neighbors import KNeighborsClassifier
#Running KNN model for k=3

knn = KNeighborsClassifier(n_neighbors = 3)

#fitting the model in train data

knn.fit(X_train, y_train)

#predicting the model with k=3

knn_pred = knn.predict(X_test)

#printing the accuracy

print(accuracy_score(y_test, knn_pred))

#Running KNN model for k=5

knn = KNeighborsClassifier(n_neighbors = 5)

#fitting the model in train data

knn.fit(X_train, y_train)

#predicting the model with k=5

knn_pred = knn.predict(X_test)

#printing the accuracy

print(accuracy_score(y_test, knn_pred))

#Running KNN model for k=9

knn = KNeighborsClassifier(n_neighbors = 9)

#fitting the model in train data

knn.fit(X_train, y_train)

#predicting the model with k=9

knn_pred = knn.predict(X_test)

#printing the accuracy

print(accuracy_score(y_test, knn_pred))
conf_mat1 = confusion_matrix(y_test, knn_pred)
conf_mat1
#confusion matrix with heatmap
plt.figure(figsize = (9,7))
sns.heatmap(conf_mat1, annot=True,cmap='Blues', fmt='g')
#importing naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
#predicting the values
nb_predict = clf.predict(X_test)
nb_predict
#printing the accuracy
accuracy_score(y_test, nb_predict, normalize = True)
conf_mat2 = confusion_matrix(y_test, nb_predict)
conf_mat2
#confusion matrix with heatmap
plt.figure(figsize = (9,7))
sns.heatmap(conf_mat1, annot=True,cmap='Blues', fmt='g')
from sklearn.svm import SVC
svc_model = SVC(C= .1, kernel='linear', gamma= 1)
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)
prediction
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
#confusion matrix with heatmap
conf_mat3 = confusion_matrix(y_test, prediction)
sns.heatmap(conf_mat3, annot=True,cmap='Blues', fmt='g')