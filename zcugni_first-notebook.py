import numpy as np

import pandas as pd



data_train = pd.read_csv("../input/train.csv")

#prediction from these will be submitted

data_val = pd.read_csv("../input/test.csv")



#First glance at the data.

data_train.sample(10)

#Removing the useless columns

to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]

data_train = data_train.drop(to_drop, axis=1)

data_val = data_val.drop(to_drop, axis=1)



#Checking data for nulls

#defining a function to re-use

def check_null(dataframe):

    null_count = dataframe.isnull().sum()

    for index, val in enumerate(null_count):

        if (val != 0):

            print(null_count.index.values[index], ":", val)



            

print("train set")

check_null(data_train)

print("-" * 15)

print("validation set")

check_null(data_val)
# Filling null values

data_train["Embarked"].fillna(data_train["Embarked"].mode()[0], inplace = True)

data_train["Age"].fillna(data_train["Age"].median(), inplace = True)

data_val["Age"].fillna(data_val["Age"].median(), inplace = True)

data_val['Fare'].fillna(data_val['Fare'].mean(), inplace = True)



# Checking data for null again

print("train set")

check_null(data_train)

print("-" * 15)

print("validation set")

check_null(data_val)
# Checking that the values are coherent

print(data_train["Survived"].unique())

print(data_train["Pclass"].unique())

print(data_train["Sex"].unique())

print(data_train["SibSp"].unique())

print(data_train["Parch"].unique())

print(data_train["Embarked"].unique())

print(data_train["Age"].max())

print(data_train["Age"].min())

print(data_train["Fare"].max())

print(data_train["Fare"].min())

#They are, so we won't do anything
#Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder

data_train["Sex"] = LabelEncoder().fit_transform(data_train["Sex"])

data_val["Sex"] = LabelEncoder().fit_transform(data_val["Sex"])



#Creating dummies and then dropping 1 for the dummy variable trap

dummies = pd.get_dummies(data_train["Embarked"]).drop("C", axis=1)

data_train = data_train.drop("Embarked", axis=1)

data_train = data_train.join(dummies)



#Doing the same for the test data

dummies = pd.get_dummies(data_val["Embarked"]).drop("C", axis=1)

data_val = data_val.drop("Embarked", axis=1)

data_val = data_val.join(dummies)



#Removing survived column of train data

y_train = data_train["Survived"]

data_train = data_train.drop("Survived", axis=1)

data_train.sample(10)



print(data_train.sample(10))

print(data_val.sample(10))
#Merging SibSp and Parch into a new feature : FamilySize

data_train["FamilySize"] = data_train["SibSp"] + data_train["Parch"]

data_train = data_train.drop(["SibSp", "Parch"], axis=1)



data_val["FamilySize"] = data_val["SibSp"] + data_val["Parch"]

data_val = data_val.drop(["SibSp", "Parch"], axis=1)



print(data_train.sample(10))

print(data_val.sample(10))
from sklearn.preprocessing import StandardScaler

subset_train = StandardScaler().fit_transform(data_train[["Age", "Fare"]])

data_train_scale = np.hstack((data_train.iloc[:, 0:2].values, data_train.iloc[:, 4:].values))

data_train_scale = np.hstack((data_train_scale, subset_train))



subset_val = StandardScaler().fit_transform(data_val[["Age", "Fare"]])

data_val_scale = np.hstack((data_val.iloc[:, 0:2].values, data_val.iloc[:, 4:].values))

data_val_scale = np.hstack((data_val_scale, subset_val))

#Training different model

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



model = LogisticRegression(penalty="l1", solver="liblinear")

model.fit(data_train, y_train)

accuracies = cross_val_score(estimator = model, X = data_train, y = y_train, cv = 10)

print("Logistic regression:", accuracies.mean())



model = RandomForestClassifier(criterion="entropy", n_estimators=100, min_samples_split=11)

model.fit(data_train, y_train)

accuracies = cross_val_score(estimator = model, X = data_train, y = y_train, cv = 10)

print("Random forest:", accuracies.mean())



model = XGBClassifier()

model.fit(data_train, y_train)

accuracies = cross_val_score(estimator = model, X = data_train, y = y_train, cv = 10)

print("XGBoost:", accuracies.mean())



model = SVC(kernel="linear", gamma="scale")

model.fit(data_train_scale, y_train)

accuracies = cross_val_score(estimator = model, X = data_train_scale, y = y_train, cv = 10)

print("SVC:", accuracies.mean())



model = KNeighborsClassifier(algorithm="ball_tree", n_neighbors=10, p =1, weights="distance")

model.fit(data_train_scale, y_train)

accuracies = cross_val_score(estimator = model, X = data_train_scale, y = y_train, cv = 10)

print("K-Nearest-Neighbors:", accuracies.mean())



model = GaussianNB()

model.fit(data_train, y_train)

accuracies = cross_val_score(estimator = model, X = data_train, y = y_train, cv = 10)

print("Naive bayes :", accuracies.mean())
model = RandomForestClassifier(criterion="entropy", n_estimators=100, min_samples_split=11)

model.fit(data_train, y_train)



y_pred = model.predict(data_val).reshape(418, 1)

#adding the index

indexes = np.empty([418, 1])

passenger_id = 892

index = 0

for val in indexes:

    indexes[index] = passenger_id

    index += 1

    passenger_id += 1

result = np.hstack((indexes, y_pred)).astype(int)

df = pd.DataFrame(result)

df.columns = ["PassengerId","Survived"]

df.to_csv("output.csv", index=False)