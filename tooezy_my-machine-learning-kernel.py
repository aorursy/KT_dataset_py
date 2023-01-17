# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data["class"].unique()
data["class"].value_counts()
data.info()
Normal_data = data[data["class"] == "Normal"]
Abnormal_data = data[data["class"] == "Abnormal"]
plt.scatter("pelvic_radius","lumbar_lordosis_angle", data = Normal_data, color= "green",label="Normal", alpha= 0.6  )
plt.scatter("pelvic_radius","lumbar_lordosis_angle", data = Abnormal_data, color= "red" ,label = "Abnormal", alpha= 0.7 )
plt.xlabel("pelvic_radius")
plt.ylabel("lumbar_lordosis_angle")
plt.legend()
plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       edgecolor= "black")
plt.show()
data["class"] = [1 if row == "Normal" else 0 for row in data["class"]]
y_data = data["class"].values
x_data = data.drop(["class"], axis = 1)

#normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_data, test_size = 0.3, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
print("Score", knn.score(x_test, y_test))
knn_acc_list = []
for neighbour in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors=neighbour)
    knn2.fit(x_train, y_train)
    knn_acc_list.append(knn2.score(x_test, y_test))
plt.figure(figsize=(12,8))
plt.plot(range(1,30), knn_acc_list )
plt.xlabel("Neighbour")
plt.ylabel("Accuracy")
plt.show()
from sklearn.svm import SVC
svm = SVC(random_state = 42)
svm.fit(x_train, y_train)
svm_score = svm.score(x_test,y_test)
print("Accuracy of svm: ",svm_score)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_score = nb.score(x_test,y_test)
print("Naive bayes score: ", nb_score)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_score = dt.score(x_test, y_test)
print("Decision tree socore: ", dt_score)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train, y_train)
rf_score = rf.score(x_test, y_test)
print("Random forest score :", rf_score)
rf_acc_list = []
estimator_list = range(10, 200, 10)
for i in estimator_list:
    rf2 = RandomForestClassifier(n_estimators= i)
    rf2.fit(x_train, y_train)
    rf_acc_list.append(rf2.score(x_test, y_test))
plt.figure(figsize=(12,8))
plt.plot(estimator_list, rf_acc_list)
plt.xlabel("Tree count")
plt.ylabel("Accuracy")
plt.show()