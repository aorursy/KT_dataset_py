# always include aliased modules

# these are the most common ones that we'll use in our Big Data class

import math as m

import numpy as np

import scipy as sp

import pandas as pd

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns  #IMPORTANT - UPGRADE SEABORN TO VERSION 0.9.0 IN ANACONDA ENV

import statsmodels.api as sm

import statsmodels.formula.api as smf
# makes the graphs visible in this notebook

%matplotlib inline  
# apply the default seaborn theme, scaling, and color palette

sns.set()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

#the code below displays a barplot

#barplots are helpful because they show me a lot about 



ax = sns.barplot(x="Pclass", y="Survived", data = train_data )
# the code below graphs a scatter plot.

#Note: I had to do a lot of outside research in order to find the code that worked for my version on my computer.



sns.regplot(x= "Pclass", y = "Fare", data = train_data)
#For the code below I want to drop some all of the collumns with the names, ticket, fare, cabin, and Embarked.

train_data = train_data.drop("Ticket",axis=1)

    

train_data = train_data.drop("Fare",axis=1)



train_data = train_data.drop("Cabin",axis=1)



train_data = train_data.drop("Embarked",axis=1)



#Then I want to print the first 9 rows to make sure that my code worked

train_data.head(9)
train_data.info()
#For the code below I want to drop some all of the collumns with the name Age.

train_data = train_data.drop("Age",axis=1)



#Then I want to print the first 9 rows to make sure that my code worked

train_data.head(9)

    
#the code below displays a barplot

#barplots are helpful because they show me a lot about 



ax = sns.barplot(x="Sex", y="Survived", data = train_data )
#

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
#

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
#the code below displays a barplot

#barplots are helpful because they show me a lot about 



ax = sns.barplot(x="SibSp", y="Survived", data = train_data )
#the code below displays a barplot

#barplots are helpful because they show me a lot about 



ax = sns.barplot(x="Parch", y="Survived", data = train_data )
#For the code below I want to drop some all of the collumns with the name Parch, looking at the number of parents and children you were traveling with.

train_data = train_data.drop("Parch",axis=1)



#For the code below I want to drop some all of the collumns with the name SibSp, looking at number of siblings.

train_data = train_data.drop("SibSp",axis=1)



#Then I want to print the first 9 rows to make sure that my code worked

train_data.head(9)
#For the code below I want to drop some all of the collumns with the name "Name", looking at each individual name on the titanic.

train_data = train_data.drop("Name",axis=1)



#Then I want to print the first 9 rows to make sure that my code worked

train_data.head(9)





test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.info()
#

from sklearn.ensemble import RandomForestClassifier



#y means the labels

y = train_data["Survived"]



#features I want to use to train the data

features = ["Pclass", "Sex"]



#training features

X = pd.get_dummies(train_data[features])



#Testing features

X_test = pd.get_dummies(test_data[features])



#Creates a new empty model called model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

#fills the model with data in order to train it

model.fit(X, y)

#Creates an emty list called predictions using test data.

predictions = model.predict(X_test)

#Creates dataframe with just passanger ID and Survived collumns

output_randomF = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})





# K-Nearest Neighbors classifier

from sklearn.neighbors import KNeighborsClassifier



#make empty classifier

knn_classifier = KNeighborsClassifier()



#fills the model with data in order to train it

knn_classifier.fit(X, y)



#Creates an emty list called predictions using test data.

knn_predictions = knn_classifier.predict(X_test)



#Creates dataframe with just passanger ID and Survived collumns 

output_knn = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': knn_predictions})







from sklearn.svm import SVC



#make empty classifier

support_classifier = SVC()



#fills the model with data in order to train it

support_classifier.fit(X, y)



#

support_predictions = support_classifier.predict(X_test)



#Creates dataframe with just passanger ID and Survived collumns

output_support = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': support_predictions})









# imports tree

from sklearn import tree



# does a decision tree classifier

tree_classifier = tree.DecisionTreeClassifier()



#fills the model with data in order to train it

tree_classifier.fit(X, y)



#

tree_predictions = tree_classifier.predict (X_test)



#Creates dataframe with just passanger ID and Survived collumns

output_tree = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': tree_predictions})







output_knn.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")


