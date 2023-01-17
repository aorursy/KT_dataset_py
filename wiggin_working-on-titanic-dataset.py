5#All of below were imported because they were in the first titanic tutorial

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

from collections import Counter





# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

#reading data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
#Looking at the variables we have

#'PassengerId' 

#'Survived' - [0,1]

#'Pclass' - ticket class [1,2,3]

#'Name' - presumably name, but should discard this?

#'Sex' 

#'Age' - fractional if less than one, estimated values have format xx.5 

#'SibSp' - number of siblings or spouses on board

#'Parch' - number of parents or children on board

#'Ticket' - ticket number, non-informative

#'Fare' - passenger fare (should depend on pclass no?)

#'Cabin' - cabin number - may be informative

#'Embarked' - port of embarkation [C,Q,S]



## advice: separate into categorical or numerical

#Categorical: Survived, Pclass, Embarked,Sex,FamilyOnBoard(combined)

#Numerical: Age,Fare,SibSp,Parch,Cabin



#things that could be combined in a reasonable way

#family on board (SibSp>1, Parch>1)

#ChildrenInvolved(parch>1, age < 14)

#mothers

#fathers
print(train_df.columns.values)
#name feature contains titles, which could be informative, but we are going to reduce them to a binary "rare" value

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
combine = [train_df, test_df]
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'RareM')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Dona'], 'Mrs')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "RareM": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#now converting all values to numerical instead of text

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#now filling in age values by guessing using pclass and gender combinations.

#note probably will want to change this to use different ranges based on title and child status



guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1) & (dataset.Title != 4),\

                    'Age'] = guess_ages[i,j]

    

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == 4), 'Age']=10 #masters are children, not average male age.

    

    dataset['Age'] = dataset['Age'].astype(int)

                                                                                                 

combine = [train_df, test_df]

train_df.head()
for dataset in combine:    

    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

combine = [train_df, test_df]

train_df.head()



train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#author wants to drop all of the family info in favor of "isalone" which seems questionable to me as this almost certainly matters a lot.

for dataset in combine:

    dataset['IsAlone'] = 0

    #dataset['BigFamily'] = 0 I think this leads to overfitting

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    #dataset.loc[dataset['FamilySize'] >= 5, 'BigFamily'] = 1



#train_df[['BigFamily', 'Survived']].groupby(['BigFamily'], as_index=False).mean()
#combining these also is questionable to me.

for dataset in combine:

    dataset['Age*Class'] = (dataset.Age+1) * dataset.Pclass #added this +1 because class matters even for kids



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



#now everything is numeric
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#want to divide fare by number of people using that ticket

train_counter = Counter(train_df['Ticket'])

for tick in train_df['Ticket'].unique():

    train_df.loc[train_df['Ticket']==tick,'n_on_ticket'] = train_counter[tick]

    

test_counter = Counter(test_df['Ticket'])

for tick in test_df['Ticket'].unique():

    test_df.loc[test_df['Ticket']==tick,'n_on_ticket'] = test_counter[tick]



combine = [train_df,test_df]
for dataset in combine:

    dataset['Fare'] = dataset.Fare/dataset.n_on_ticket



train_df.head()
#I don't really understand why we want everything in int format but w/e

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.76, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.76) & (dataset['Fare'] <= 8.85), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 8.85) & (dataset['Fare'] <= 24.288), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 24.288, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



combine = [train_df, test_df]


train_df = train_df.drop(['PassengerId','Ticket', 'Cabin','Name','SibSp', 'FamilySize','Parch','n_on_ticket'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin','Name','SibSp', 'FamilySize','Parch','n_on_ticket'], axis=1)
train_df.info()

train_df = train_df.drop('FareBand',axis=1)
test_df.info()
#Now we're FINALY READY TO MODEL STUFF



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

#Y_pred

submission.to_csv('submission.csv',index=False)