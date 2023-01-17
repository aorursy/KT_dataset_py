%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, ElasticNet

from sklearn.naive_bayes import BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

plt.rcParams['figure.figsize'] = [18, 9]
train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col=0)

test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)
train.head()
scatter_matrix(train, diagonal='hist')
train.corr()
Y = train["Survived"].values
# Drop columns name, ticket and cabin

train = train.drop(columns=["Name", "Ticket", "Cabin"])
# Convert sex to categorical feature

one_hot_sex = OneHotEncoder(sparse=False)

sex_encoded = one_hot_sex.fit_transform(train["Sex"].values.reshape(-1, 1))

X_categorical = sex_encoded
train["Age"].describe()

train["Age"].hist()

train["Age"].isnull().sum()



# Assign average of ages tom all Nan values

mean_age = round(train["Age"].mean())

train["Age"].loc[train["Age"].isnull()] = mean_age



X_numerical = train["Age"].values.reshape(-1, 1)
train["Pclass"].describe()

train["Pclass"].hist()

train["Pclass"].isnull().sum()



# Pclass to categorical

one_hot_pclass = OneHotEncoder(sparse=False)

pclass_encoded = one_hot_pclass.fit_transform(train["Pclass"].values.reshape(-1, 1))

X_categorical = np.hstack((X_categorical, pclass_encoded))
train["SibSp"].isnull().sum()

train["SibSp"].hist()



# Add raw value of SibSp to features

X_numerical = np.hstack((X_numerical, train["SibSp"].values.reshape(-1, 1)))
train["Parch"].isnull().sum()

train["Parch"].hist()



# Add raw value of Parch to features

X_numerical = np.hstack((X_numerical, train["Parch"].values.reshape(-1, 1)))
train["Fare"].isnull().sum()

(train["Fare"]<0).sum()

train["Fare"].hist(bins=100)



# Add raw value of Fare to features

X_numerical = np.hstack((X_numerical, train["Fare"].values.reshape(-1, 1)))
train["Embarked"].isnull().sum()



# To Nan values give most common category.

train["Embarked"].loc[train["Embarked"].isnull()] = train["Embarked"].value_counts().idxmax()

train["Embarked"].isnull().sum()

train["Embarked"].hist()



# Embarked to categorical

one_hot_embarked = OneHotEncoder(sparse=False)

embarked_encoded = one_hot_embarked.fit_transform(train["Embarked"].values.reshape(-1, 1))

X_categorical = np.hstack((X_categorical, embarked_encoded))
train.head()
scaler = StandardScaler()

X_numerical = scaler.fit_transform(X_numerical)

X = np.hstack((X_numerical, X_categorical))
kf = KFold(20, shuffle=True, random_state=41)



classifier = GradientBoostingClassifier()



classifier_accuracy = []

for train_index, test_index in kf.split(train):

    train_X, train_Y = X[train_index], Y[train_index]

    test_X, test_Y = X[test_index], Y[test_index]

    classifier.fit(train_X, train_Y)

    predictions = classifier.predict(test_X)

    accuracy = accuracy_score(test_Y, predictions)

    classifier_accuracy.append(accuracy)



print(sum(classifier_accuracy)/len(classifier_accuracy))
test_sex_encoded = one_hot_sex.transform(test["Sex"].values.reshape(-1, 1))

test["Age"].loc[test["Age"].isnull()] = mean_age

test_pclass_encoded = one_hot_pclass.transform(test["Pclass"].values.reshape(-1, 1))

test["Embarked"].loc[test["Embarked"].isnull()] = train["Embarked"].value_counts().idxmax()

test_embarked_encoded = one_hot_embarked.transform(test["Embarked"].values.reshape(-1, 1))

test["Fare"].loc[test["Fare"].isnull()] = train["Fare"].mean()



test_X_categorical = np.hstack((test_sex_encoded, test_pclass_encoded, test_embarked_encoded))

test_X_numerical = np.hstack((test["Age"].values.reshape(-1, 1), test["SibSp"].values.reshape(-1, 1), test["Parch"].values.reshape(-1, 1), test["Fare"].values.reshape(-1, 1)))



test_X_numerical = scaler.transform(test_X_numerical)

test_X = np.hstack((test_X_numerical, test_X_categorical))
classifier.fit(train_X, train_Y)

predictions = classifier.predict(test_X)
data = []

for (passenger_id, passenger_data), prediction in zip(test.iterrows(), predictions):

    data.append([passenger_id, prediction])



series = pd.DataFrame(data, columns=["PassengerID", "Survived"]).set_index("PassengerID")

series.to_csv("predictions.csv")