#load necessary modules
import numpy as np 
import pandas as pd 
from sklearn import tree
import seaborn as sns

import os
print(os.listdir("../input"))


#load the train and test data
traind=pd.read_csv("../input/train.csv")
testd=pd.read_csv("../input/test.csv")

#view the data ....head and tail
traind.head()
traind.tail()
#summary of all the data to check for any nulls, data types
traind.info()
#Age, Cabin, and Embarked have nulls
#Drop Cabin-is missing alot. . may be missing important information
#Drop PassengerId - merely a continuous number with probably no much to learn from
testd.info()
traind.describe()
#age missing 205
#numeric variables - passengerId,Survived,Pclass,Age,SibSp,Parch,Fare

#train data shows 38% survived while 62% did not

sns.countplot(traind['Survived'])
traind['Survived'].value_counts(normalize=True)
#there was a lower chance of survival in 3rd class wich had lower socio-economic people

sns.countplot(traind['Pclass'], hue=traind['Survived'])
#Name ?
#There seems to be some form of a relationship between rate of survival and sex. .more men 
#perished than those that survived.
#more women survived compered to those that died.
sns.countplot(traind['Sex'], hue=traind['Survived'])
#Age
#fix the missing data to get a better picture of the variable? maybe later
#the ages seems to be relatively evenly distributed same are their survival rate
#A higher survival rate for the younger age group 0.4-19, however no clear correlation 

pd.qcut(traind['Age'],5).value_counts()

sns.countplot (pd.qcut(traind['Age'],5), hue=traind['Survived'])
traind['Survived'].groupby(pd.qcut(traind['Age'],5)).mean()
#possible a high number of siblings affected survival rate?
traind['SibSp'].value_counts()
traind['Survived'].groupby(traind['SibSp']).mean()
#parc
#possible a higher number of children reduced chances of survival
traind['Parch'].value_counts()

traind['Survived'].groupby(traind['Parch']).mean()
#Tickets...ticket number could probably be an indication of how much paid, class, cabin one was in
# and thus a clear indication of the chances of survival
# The numbers are different...further analysis may reveal the meaning and implications there of
traind['Ticket'].head(30)
# ther is an increase in the survival mean as the fare increases
pd.qcut(traind['Fare'],5).value_counts()
traind['Survived'].groupby(pd.qcut(traind['Fare'],5)).mean()
#most people embarked at Southampton, survival rate of those that embarked in cherbourg was higher
sns.countplot(traind['Embarked'], hue=traind['Survived'])

#drop Ticket, Cabin, PassengerId
print("Before", traind.shape, testd.shape, )

traind = traind.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
testd = testd.drop(['Ticket', 'Cabin', ], axis=1)
combine = [traind, testd]

"After", traind.shape, testd.shape, combine[0].shape, combine[1].shape
# convert sex to numeric
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

traind.head()
#replace the missing values in the Embarked feature with S
traind = traind.fillna({"Embarked": "S"})
testd = testd.fillna({"Embarked": "S"})

#convert Embarked to numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
traind['Embarked'] = traind['Embarked'].map(embarked_mapping)
testd['Embarked'] = testd['Embarked'].map(embarked_mapping)

traind.head()
traind.describe()
testd.describe()
#replace the missing values in the Age with the age means train and test respectively
traind = traind.fillna({"Age": 29})
testd = testd.fillna({"Age": 30})

#replace the missing values in the test (fare) with the fare mean 
testd = testd.fillna({"Fare": 35})
testd.info()
#Map age into 5 numeric groups
traind['Agegroup']= pd.qcut(traind['Age'], 5, labels = [1, 2, 3, 4, 5])
testd['Agegroup']= pd.qcut(testd['Age'], 5, labels = [1, 2, 3, 4, 5] )                      
#map fare into 5 numeric groups
traind['Faregroup'] = pd.qcut(traind['Fare'], 5, labels = [1, 2, 3, 4, 5])
testd['Faregroup'] = pd.qcut(testd['Fare'], 5, labels = [1, 2, 3, 4, 5])

traind = traind.drop(['Age', 'Fare', 'Name' ], axis=1)
testd = testd.drop(['Age', 'Fare', 'Name' ], axis=1)
traind.head()
testd.head()

from sklearn.model_selection import train_test_split

# identify the features (x) and the target (y)
X = traind.drop(['Survived'], axis=1)
y = traind['Survived']

# randomly split the training data to test the model 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print(round(acc_decision_tree,2,), "%")


# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print(round(acc_random_forest,2,), "%")


#view the test data
testd.head()
# predict with random forest
submission = pd.DataFrame({
    "PassengerId" : testd['PassengerId'],
    "Survived" : random_forest.predict(testd.drop('PassengerId', axis=1))
})
submission.head(8)

submission.info()
submission.to_csv('submission.csv', index=False)