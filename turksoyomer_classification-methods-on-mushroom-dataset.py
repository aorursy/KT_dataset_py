# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.info()
data["class"].unique()
# Changing class values to "1" and "0"s.

data["class"] = [1 if i == "p" else 0 for i in data["class"]]



# Dropping "veil-type" column.

data.drop("veil-type",axis=1,inplace=True)
for column in data.drop(["class"], axis=1).columns:

    value = 0

    step = 1/(len(data[column].unique())-1)

    for i in data[column].unique():

        data[column] = [value if letter == i else letter for letter in data[column]]

        value += step
data_check = data.head()

data_check = data_check.append(data.tail())

data_check
from sklearn.model_selection import train_test_split

y = data["class"].values    # "class" column as numpy array.

x = data.drop(["class"], axis=1).values    # All data except "class" column. I didn't use normalization because all data has values between 0 and 1.

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)    # Split data for train and test.
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="lbfgs")

lr.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))
from sklearn.neighbors import KNeighborsClassifier

best_Kvalue = 0

best_score = 0

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    if knn.score(x_test,y_test) > best_score:

        best_score = knn.score(x_train,y_train)

        best_Kvalue = i

print("""Best KNN Value: {}

Test Accuracy: {}%""".format(best_Kvalue, round(best_score*100,2)))
from sklearn.svm import SVC

svm = SVC(random_state=42, gamma="auto")

svm.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(svm.score(x_test,y_test)*100,2)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(dt.score(x_test,y_test)*100,2)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(x_train,y_train)

print("Test Accuracy: {}%".format(round(rf.score(x_test,y_test)*100,2)))
from sklearn.metrics import confusion_matrix

y_pred_lr = lr.predict(x_test)

y_true_lr = y_test

cm = confusion_matrix(y_true_lr, y_pred_lr)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred_lr")

plt.ylabel("y_true_lr")

plt.show()
y_pred_rf = rf.predict(x_test)

y_true_rf = y_test

cm = confusion_matrix(y_true_rf, y_pred_rf)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred_rf")

plt.ylabel("y_true_rf")

plt.show()