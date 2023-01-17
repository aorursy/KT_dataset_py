# Import the libraries

import numpy as np

import pandas as pd

from sklearn import tree





# Load the train and test datasets to create two DataFrames

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Cleaning and Formatting the Data

train["Age"] = train["Age"].fillna(train["Age"].median())



# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

# Creating the decision tree



# Create the target and features numpy arrays: target, features_one

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))

# Check test["Fare"] for missing values

test[test['Fare'].isnull()]
# Check test["Age"] for missing values

test[test['Age'].isnull()]
# Impute the missing value with the median

test.Fare[152] = test.Fare.median()

test["Age"][test["Age"].isnull()] = test["Age"].median()



# Convert the male and female groups to integer form

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1
# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values



# Make your prediction using the test set and print them.

my_prediction = my_tree_one.predict(test_features)

print(my_prediction)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

# Create new arrays with added features

features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values



# Impute the Embarked variable

test["Embarked"] = test["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



test_features_two = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values



#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)



my_solution_two = pd.DataFrame(my_tree_two.predict(test_features_two), PassengerId, columns = ["Survived"])

my_solution_two.to_csv("my_solution_two.csv", index_label = ["PassengerId"])



#Print the score of the new decison tree

my_tree_two.score(features_two, target)
# Create train_two with the newly defined feature

train_two = train.copy()

train_two["family_size"] = 1 + train_two["SibSp"] + train_two["Parch"]



# Create a new feature set and add the new feature

features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values



test_two = test.copy()

test_two["family_size"] = 1 + test_two["SibSp"] + test_two["Parch"]



# Create a new feature set and add the new feature

test_features_three = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values



# Define the tree classifier, then fit the model

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three, target)



my_solution_three = pd.DataFrame(my_tree_three.predict(test_features_three), PassengerId, columns = ["Survived"])

my_solution_three.to_csv("my_solution_three.csv", index_label = ["PassengerId"])



# Print the score of this decision tree

my_tree_three.score(features_three, target)