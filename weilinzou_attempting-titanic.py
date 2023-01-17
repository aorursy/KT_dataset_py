import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # pretty visualizations

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#First I'm going to get the files as a DataFrame
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#We'll need this later to submit the result
test2 = test.copy()
full_data = [train,test]
#Let's see the data that we're working with.
train.head()
#How much data do we have?
train.shape
# Lets see what we have to work with
train.columns
#We can reality check the data to see if it makes sense. The average age with 29 years old, the youngest person  was less than a year old and the
#oldest is 80 years old. The fare price varied drammatically from the minimum fare of free to 512. We can also see what data is missing
train.describe()
#We can see that we only have very little data for "Cabin"
train.describe(include="O")
#Lets see which variables have an impact on survival so we can put them into our model
train[['Pclass','Survived']].groupby(['Pclass']).mean()
#We can see class is highly correlated to survival so we should probably include that in our model
#What about sex?
train[['Sex','Survived']].groupby(['Sex']).mean()
#We can see this is probably one of the most important determinants to surival so we'll include that too
#'SibSp' and 'Parch' refer to the siblings/spouses and parents/children on the ship respectively. We're not missing
#Any data for those two variables so lets make a new variable called "FamilySize"
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train[['FamilySize','Survived']].groupby(['FamilySize']).mean()
#It looks to be pretty correlated, but I'm not entirely sure about the direction the correlation is going in or how much randomness is a
#factor in these variables since the survival for FamilySize ==4 is over 70% whereas it drops to around 15-30% for those over 5
# Lets make a new variable "IsAlone" based on if the familysize is one or not
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone','Survived']].groupby(['IsAlone']).mean()
#This one seems pretty good
#Lets look at the next one, which is 'Age', which would make sense if it is correlated with survival
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
#Age looks pretty correlated with survival, esepcially for children, who are more likely to survive, and the elderly, who may not be.
#This one seems pretty good
#Lets look at the next one, which is 'Age', which would make sense if it is correlated with survival
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
#It definitely looks like higher fares are associated with survival, although it is a bit harder to tell at the tail where the graph is skewed.
#Let's break it down into four segments and see if we can see it more clearly that way
train['FareCategories'] = pd.qcut(train['Fare'],4)
train[['FareCategories','Survived']].groupby(['FareCategories']).mean()
#We can definitely see an increase in survival as the fare increases
#Not sure what we can do with Cabin and Ticket so lets skip those for now and go to Embarked
train[['Embarked','Survived']].groupby(['Embarked']).mean()
#It definitely looks correlated, but logically it doesn't really make sense to me unless it's because it's also correlated with one of our
#other factors somehow
#Now that we have the factors that we want to include, lets drop the columns that we don't need anymore
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize','Embarked']
#We need to also to remember to drop elements from 'FareCategories' since we made that column earlier to visualize our data a little better
train = train.drop(drop_elements, axis = 1)
train = train.drop(['FareCategories'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
#We still have many missing values in our "Age" category 
#Let's fill the missing values with the mean value for Age
train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train['Age'].describe()
#We have one missing value for the "Fare" column for the test column so we'll fill it in with the mean()
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} )
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} )

#Our dataset looks good for running our machine learning models on now
X = train.drop("Survived", axis = 1)
Y = train["Survived"]
#Lets try out these machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# going to split the data to avoid overfitting so we can pick the best model
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

logreg.score(X_test, Y_test)
random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

random_forest.score(X_test, Y_test)
svc = SVC()

svc.fit(X_train, Y_train)

svc.score(X_test, Y_test)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

knn.score(X_test, Y_test)
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

gaussian.score(X_test, Y_test)
#I think we'll want to use the random forest model, so we'll retrain it using the entire training set, run it on our test set and submit it!
random_forest_real = RandomForestClassifier()
random_forest_real.fit(X, Y)
Y_pred = random_forest_real.predict(test)

submission = pd.DataFrame({
        "PassengerId": test2["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_submission.csv', index=False)