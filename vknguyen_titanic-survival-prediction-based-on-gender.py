import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
#Import the train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#View the data
print(train.head())
print(test.head())

#Understand data
print(train.shape)
print(test.shape)

print(list(train))
print(list(test))
#Look at the survived and not survived passengers
print('0 is not survived and 1 is survived:')
print(train["Survived"].value_counts())

print("Proportion of the fallens and the surviors:")
print(train["Survived"].value_counts(normalize = True))
      
#Look at males survival
print("The number of males who survived-1 and those who did not survive-0:")
print(train["Survived"][train["Sex"] == 'male'].value_counts())
print("The proportion of fallen males and survived males:")
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

#Look at females survival
print("The number of females who survived-1 and those who did not survive-0:")
print(train["Survived"][train["Sex"] == 'female'].value_counts())
print("The proportion of fallen females and survived females:")
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
      
#Look at the children
#We create a new column that defines 1 - children under 18 and 0 - people 18 or older.
train["Child"] = float('NaN')
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

print(train["Child"].head())
print(train["Child"].tail())

print("Survival rate of children under 18:")
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print("Survival rate of passengers 18 or older:")
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
#We want to test if female survived while male did not
test_copy = test
test_copy["Survived"] = 0

test_copy["Survived"][test_copy["Sex"] == 'female'] = 1
print("Prediction of survival in test dataset:")

prediction = test_copy[["PassengerId", "Survived"]]
print(prediction.head())
prediction.to_csv('prediction.csv', index = False, index_label = False)
