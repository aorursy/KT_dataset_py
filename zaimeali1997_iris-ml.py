import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input/iris"]).decode("utf8"))
data = pd.read_csv('../input/iris/Iris.csv', index_col='Id')
data.head()
# to check how many species are in a dataset

data.Species.unique()
# there are 3 types of Species

data.Species.value_counts()
print("Length of Dataset: ", len(data))

print('Shape of Dataset: ', data.shape)
data.info()
data.notnull().sum()
# another way to check null values 

data.isna().sum()
fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter', color='green', x=['SepalLengthCm'], y=['SepalWidthCm'], label='Setosa', figsize=(12, 6))

data[data.Species == 'Iris-virginica'].plot(kind='scatter', color='orange', x='SepalLengthCm', y='SepalWidthCm', label='Virginica', ax=fig)

data[data.Species == 'Iris-versicolor'].plot(kind='scatter', color='blue', x='SepalLengthCm', y='SepalWidthCm', ax=fig, label='Versicolor')

plt.grid()

fig.set_ylabel('Sepal Width')

fig.set_xlabel('Sepal Length')

plt.title('Length vs Width')

plt.show()
fig = data[data.Species == 'Iris-setosa'].plot(kind='scatter', color='green', x='PetalLengthCm', y='PetalWidthCm', label='Setosa', figsize=(12, 6))

data[data.Species == 'Iris-virginica'].plot(kind='scatter', color='orange', x='PetalLengthCm', y='PetalWidthCm', label='Virginica', ax=fig)

data[data.Species == 'Iris-versicolor'].plot(kind='scatter', color='blue', x='PetalLengthCm', y='PetalWidthCm', ax=fig, label='Versicolor')

plt.grid()

fig.set_ylabel('Petal Width')

fig.set_xlabel('Petal Length')

plt.title('Length vs Width')

plt.show()
data.hist(edgecolor='black', figsize=(12, 6))

plt.show()
data[data.Species == 'Iris-setosa'].hist(edgecolor='black', figsize=(10, 6))
data[data.Species == 'Iris-versicolor'].hist(edgecolor='black', figsize=(10, 6))
data[data.Species == 'Iris-virginica'].hist(edgecolor='black', figsize=(10, 6))
plt.figure(figsize=(10, 5))

sns.heatmap(data=data.corr(), annot=True)

plt.show()
# now we know this is a classification problem 

from sklearn.model_selection import train_test_split



X = data.iloc[:, :-1]

y = data.iloc[:, -1]
X.head()
y.head()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
print(train_X.shape)

print(test_X.shape)
# Classification Algos are:

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
# LogisticRegression on all features

lr = LogisticRegression()

lr.fit(train_X, train_y)

pred_lr = lr.predict(test_X)

print('Logistic Regression\n')

accScore = accuracy_score(pred_lr, test_y)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(pred_lr, test_y)

print('Confusion Matrix')

print(confMatrix)
# Decision Tree Classifier on all Features

dTree = DecisionTreeClassifier()

dTree.fit(train_X, train_y)

dt_pred = dTree.predict(test_X)

print('Decision Tree Classifier\n')

accScore = accuracy_score(dt_pred, test_y)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(dt_pred, test_y)

print('Confusion Matrix')

print(confMatrix)
# KNN Classifier on all Features

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(train_X, train_y)

knn_pred = knn.predict(test_X)

print('KNN Classifier\n')

accScore = accuracy_score(knn_pred, test_y)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(knn_pred, test_y)

print('Confusion Matrix')

print(confMatrix)
# Support Vector on all Features

svc = LinearSVC()

svc.fit(train_X, train_y)

svc_pred = svc.predict(test_X)

print('Support Vector Machine Classifier\n')

accScore = accuracy_score(svc_pred, test_y)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(svc_pred, test_y)

print('Confusion Matrix')

print(confMatrix)
data.columns
S, P = data[['SepalLengthCm', 'SepalWidthCm', 'Species']], data[['PetalLengthCm', 'PetalWidthCm', 'Species']]
print(data.head(2))

print(S.head(2))

print(P.head(2))
print(S.shape)

print(P.shape)
SX = S.iloc[:, :-1]

Sy = S.iloc[:, -1]

PX = P.iloc[:, :-1]

Py = P.iloc[:, -1]
train_Sx, test_Sx, train_yS, test_yS = train_test_split(SX, Sy, test_size=0.3)

train_Px, test_Px, train_yP, test_yP = train_test_split(PX, Py, test_size=0.3)
data.iloc[135, :]
train_Px.head()
# Logistic Regression on Sepal

lr_S = LogisticRegression()

lr_S.fit(train_Sx, train_yS)

S_lr_pred = lr_S.predict(test_Sx)

print('Logistic Classifier on Sepal features\n')

accScore = accuracy_score(S_lr_pred, test_yS)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(S_lr_pred, test_yS)

print('Confusion Matrix')

print(confMatrix)

print()



# Logistic Regression on Petal

lr_P = LogisticRegression()

lr_P.fit(train_Px, train_yP)

P_lr_pred = lr_P.predict(test_Px)

print('Logistic Classifier on Petal features\n')

accScore = accuracy_score(P_lr_pred, test_yP)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(P_lr_pred, test_yP)

print('Confusion Matrix')

print(confMatrix)
# Decision Tree Classifier on Sepal

dt_S = DecisionTreeClassifier()

dt_S.fit(train_Sx, train_yS)

S_dt_pred = dt_S.predict(test_Sx)

print('Decision Tree Classifier on Sepal features\n')

accScore = accuracy_score(S_dt_pred, test_yS)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(S_dt_pred, test_yS)

print('Confusion Matrix')

print(confMatrix)

print()



# Decision Tree Classifier on Petal

dt_P = DecisionTreeClassifier()

dt_P.fit(train_Px, train_yP)

P_dt_pred = dt_P.predict(test_Px)

print('Decision Tree Classifier on Petal features\n')

accScore = accuracy_score(P_dt_pred, test_yP)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(P_dt_pred, test_yP)

print('Confusion Matrix')

print(confMatrix)
# KNN Classifier on Sepal

k_S = KNeighborsClassifier(n_neighbors=3)

k_S.fit(train_Sx, train_yS)

S_k_pred = k_S.predict(test_Sx)

print('KNN Classifier on Sepal features\n')

accScore = accuracy_score(S_k_pred, test_yS)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(S_k_pred, test_yS)

print('Confusion Matrix')

print(confMatrix)

print()



# KNN Classifier on Petal

k_P = KNeighborsClassifier(n_neighbors=3)

k_P.fit(train_Px, train_yP)

P_k_pred = k_P.predict(test_Px)

print('KNN Classifier on Petal features\n')

accScore = accuracy_score(P_k_pred, test_yP)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(P_k_pred, test_yP)

print('Confusion Matrix')

print(confMatrix)
# SVC on Sepal

svc_S = LinearSVC()

svc_S.fit(train_Sx, train_yS)

pred_svc_S = svc_S.predict(test_Sx)

print('SVC on Sepal features\n')

accScore = accuracy_score(pred_svc_S, test_yS)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(pred_svc_S, test_yS)

print('Confusion Matrix')

print(confMatrix)

print()



# SVC on Petal

svc_P = LinearSVC()

svc_P.fit(train_Px, train_yP)

pred_svc_P = svc_P.predict(test_Px)

print('SVC on Petal features\n')

accScore = accuracy_score(pred_svc_P, test_yP)

print('Accuracy Score:', accScore)

confMatrix = confusion_matrix(pred_svc_P, test_yP)

print('Confusion Matrix')

print(confMatrix)
# help you can see our model is performing better on Petal which is proved by heat map that petal have a better correlation(+ve)