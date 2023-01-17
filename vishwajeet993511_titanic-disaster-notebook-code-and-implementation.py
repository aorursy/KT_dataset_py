# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data analysis and wrangling  this library for random numbers and data shuffling

import random as rnd

from IPython import get_ipython

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

#%matplotlib inline



# machine learning libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]





print(train_df.columns.values)



# pre visiting the data to get a glimpse of data

print(train_df.head())

print(train_df.describe())









#print(train_df[['Parch','Survived']].groupby(['Parch'],as_index = False).mean().sort_values(by = 'Survived', ascending = False))







# exploratory analysis of the problem by plotting variuos relations between features and output





g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)





grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();





grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()







grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()







print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



#print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])





# since some english abbreviations have the same meaning as the true words so grouping them togather



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()









title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



print(train_df.head())





# name and passenger id does not contribute directly to the output





train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]







for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#print (train_df.shape)    

#print(test_df.shape)









grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()





guess_ages = np.zeros((2,3))

guess_ages



# since age is one of the most important features contributing to the output so 

# we should fill the NA columns



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

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



#print(train_df.head())



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)





for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

    

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

    

#print(train_df.head())







for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



#print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))





for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    

    

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]





for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



#print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))





freq_port = train_df.Embarked.dropna().mode()[0]



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))





for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)





train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]



#print(train_df.head(10))

#print(test_df.head(10))





X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()







# finally applying machine learning model to the input data 

# we average the output by applying various machine learing models to reduce the noise









logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_logreg = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#print(acc_log)



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

#acc_svc





knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

#acc_knn





gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred_gaussian = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

#acc_gaussian





perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#acc_perceptron





linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred_linearsvc = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#acc_linear_svc







sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred_sgd = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

#acc_sgd





decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_dtree = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#acc_decision_tree





random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_rforest = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#acc_random_forest





# finally accessing the accuracy results and comparision of the models



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

print(models.sort_values(by='Score', ascending=False))



finaldf = { 'Y_pred_logreg' : Y_pred_logreg ,'Y_pred_svc' : Y_pred_svc,'Y_pred_linearsvc' : Y_pred_linearsvc,

'Y_pred_sgd': Y_pred_sgd,'Y_pred_rforest' :Y_pred_rforest,'Y_pred_dtree' : Y_pred_dtree,

'Y_pred_gaussian' :Y_pred_gaussian,'Y_pred_knn' :Y_pred_knn }



#finalmodels = pd.DataFrame(Y_pred_logreg,Y_pred_svc,Y_pred_linearsvc,Y_pred_sgd,Y_pred_rforest,Y_pred_dtree,Y_pred_gaussian,Y_pred_knn)

finalmodels = pd.DataFrame( data = finaldf )

finalmodels['sum'] =   finalmodels.sum(axis=1)

print(finalmodels.head(10))





# function for the voting systemm between the classifiers . maximum voted will be considered as the output one 





def f(row):

    if row['sum'] >= 4:

        val = 1

    else:

        val = 0

    return val

    

finalmodels['Survived'] = finalmodels.apply(f, axis=1)

print(finalmodels.head(10))





# converting to the dataframe



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": finalmodels['Survived']

    })

    

print(submission.head(10))

#print(submission.describe())





#submission.to_csv('../output/submission.csv', index=False)



#print(submission)