# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

test_labels = pd.read_csv("../input/genderclassmodel.csv")

train_features = train.drop(["Survived","Name","Ticket","Cabin","PassengerId"],axis=1)

train_labels = train["Survived"]

test_features = test.drop(["Name","Ticket","Cabin","PassengerId"],axis=1)

test_labels = test_labels["Survived"]



cleanup_nums = {"Sex":     {"female": 1, "male": 2}, "Embarked":{"S":1,"C":2,"Q":3}

                }

train_features.replace(cleanup_nums, inplace=True)

test_features.replace(cleanup_nums, inplace=True)



#print(train_features)



#print(train_features["Embarked"].value_counts())

#train_features["Age"].value_counts()

train_features = train_features.fillna({"Age": 30})

test_features = test_features.fillna({"Age": 30})



train_features = train_features.fillna({"Embarked": 1})

test_features = test_features.fillna({"Fare": 7.75})



#print(train_features["Embarked"].value_counts())

#print(train_features[train_features.isnull().any(axis=1)])

#print(test_features["Fare"].value_counts())



from sklearn import svm

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn import neighbors

from sklearn.metrics import accuracy_score

#print(test_features)

model1 = tree.DecisionTreeClassifier()

model2 = svm.SVC()

model3 = GaussianNB()

model4 = neighbors.KNeighborsClassifier()

    

model1.fit(train_features,train_labels)

model2.fit(train_features,train_labels)

model3.fit(train_features,train_labels)

model4.fit(train_features,train_labels)



#print(test_features[test_features.isnull().any(axis=1)])

pred1= model1.predict(test_features)

score1 =  accuracy_score(test_labels, pred1)

print("Deceision Tree accuracy " , score1)



pred2= model2.predict(test_features)

score2 =  accuracy_score(test_labels, pred2)

print("Support vector machine accuracy " , score2)



pred3= model3.predict(test_features)

score3 =  accuracy_score(test_labels, pred3)

print("Naive Bayes accuracy " , score3)



pred4= model4.predict(test_features)

score4 =  accuracy_score(test_labels, pred4)

print("Nearest Neighbors accuracy " , score4)