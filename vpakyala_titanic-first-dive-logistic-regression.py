# Import standard libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# machine learning

from sklearn.linear_model import LogisticRegression



sns.set_style('whitegrid')

%matplotlib inline
# Load test & train data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Lets understand  the csv files

train.head()

# The problem statement is to find the whether a passenger has survived or not.

# Feature reduction - No particular method used 

# From Watching Titanic movie, anyone surviving , is nothing to do with Name,Ticket Number,Embarked,Cabin,Sex

train = train.drop(['Name','Ticket','Embarked','Cabin','Sex'],axis=1)

test = test.drop(['Name','Ticket','Embarked','Cabin','Sex'],axis=1)

# Next Step is looking for missing data using describe function

train.describe()

# Age - Count has value of 714--> Does not match with number of passsenger(891)

# Lets fill the missing age with mean. Why mean? Definetly not 0, so going with Mean. 

train["Age"].fillna(train["Age"].mean(), inplace=True)

test["Age"].fillna(test["Age"].mean(), inplace=True)

train.describe()
test.head()
# I will start to use Logistic regression as the output to be predicted is categorical ( 0 or 1)



Y_train = train["Survived"]

X_train = train.drop("Survived",axis=1)

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
print ('Accuracy on the training subset: {:.3f}'.format(logreg.score(X_train,Y_train)))
# The Accuracy is only 70% with Logistic regression. Would classifier optimization help?

# How should I proceed from here??