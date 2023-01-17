# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head(3)
# drop unnecessary cloumn

iris.drop('Id', axis=1, inplace=True)
iris.shape
iris.info()
sns.countplot(data=iris, x='Species');
iris['Species'].value_counts()
g = sns.FacetGrid(data=iris, hue='Species', size=5, aspect=1.5)

g = g.map(sns.regplot, 'SepalLengthCm', 'SepalWidthCm', fit_reg=False)

g.add_legend();
g = sns.FacetGrid(data=iris, hue='Species', size=5, aspect=1.5)

g = g.map(sns.regplot, 'PetalLengthCm', 'PetalWidthCm', fit_reg=False)

g.add_legend();
g_sepal = sns.PairGrid(data=iris, x_vars=['SepalLengthCm', 'SepalWidthCm'],

                y_vars=['Species'], height=3, aspect=1.5)

g_sepal.map(sns.violinplot, inner='quartile')



g_petal = sns.PairGrid(data=iris, x_vars=['PetalLengthCm', 'PetalWidthCm'],

                y_vars=['Species'], height=3, aspect=1.5)

g_petal.map(sns.violinplot, inner='quartile');
from sklearn import preprocessing



encoder = preprocessing.LabelEncoder()

iris['EncodedSpecies'] = encoder.fit_transform(iris['Species'])
# print mapping encoded values

dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
iris.head()
from sklearn.model_selection import train_test_split



X = iris.drop(['Species'], axis=1)

y = iris['Species']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics



NB_model = GaussianNB()

NB_model.fit(X_train, y_train)



y_predict = NB_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.svm import SVC



SVM_model = SVC(gamma=0.1)

SVM_model.fit(X_train, y_train)



y_predict = SVM_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.linear_model import LogisticRegression



Logis_model = LogisticRegression(multi_class='auto', solver='lbfgs')

Logis_model.fit(X_train, y_train)



y_predict = Logis_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.tree import DecisionTreeClassifier



DT_model = DecisionTreeClassifier()

DT_model.fit(X_train, y_train)



y_predict = DT_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.ensemble import RandomForestClassifier



RF_model = RandomForestClassifier(n_estimators=100)

RF_model.fit(X_train, y_train)



y_predict = RF_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.neighbors import KNeighborsClassifier



KNN_model = KNeighborsClassifier()

KNN_model.fit(X_train, y_train)



y_predict = KNN_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
from sklearn.ensemble import GradientBoostingClassifier



GB_model = GradientBoostingClassifier()

GB_model.fit(X_train, y_train)



y_predict = GB_model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))