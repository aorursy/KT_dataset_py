# Libraries

import matplotlib.pyplot as plt # Visualization

import seaborn as sns # Visualization

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # Train and Test Split

from sklearn.linear_model import LogisticRegression # Logistic Regression Classification

from sklearn.neighbors import KNeighborsClassifier # Knn Classification

from sklearn.svm import SVC # Support Vector Machine Classification

from sklearn.naive_bayes import GaussianNB # Naive Bayes

from sklearn.tree import DecisionTreeClassifier # Decision Tree

from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier

from sklearn.metrics import confusion_matrix # Confusion Matrix



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/heart-disease-uci/heart.csv")

data.info()
data.head()
data.describe()
data.isnull().sum()
data.target.value_counts()
sns.countplot(x="target", data=data, palette="bwr")

plt.show()
countNoDisease = len(data[data.target == 0])

countHaveDisease = len(data[data.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(data.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(data.target))*100)))
sns.countplot(x='sex', data=data, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
countFemale = len(data[data.sex == 0])

countMale = len(data[data.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(data.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(data.sex))*100)))
y = data.target.values

x_data = data.drop(["target"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)) # normalization 
# train test split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
lr = LogisticRegression()

lr.fit(x_train,y_train)

print("Score is {}".format(lr.score(x_test,y_test)))
# Confusion Martix

y_pred = lr.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

print(" {} k Score: {} ".format(3,knn.score(x_test,y_test)))
# Confusion Martix

y_pred = knn.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)
svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("Score: ",svm.score(x_test,y_test))
# Confusion Martix

y_pred = svm.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)
nb = GaussianNB()

nb.fit(x_train,y_train)

print("Score: ",nb.score(x_test,y_test))
# Confusion Martix

y_pred = nb.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Score: ", dt.score(x_test,y_test))
# Confusion Martix

y_pred = dt.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("Score: ",rf.score(x_test,y_test))
# Confusion Martix

y_pred = rf.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

print("Confusion Martix:")

print(cm)