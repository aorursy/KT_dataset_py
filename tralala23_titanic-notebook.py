# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling

# import pandas as pd

# import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.info()

print('\n' ,'='*40,'\n')

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
pclass_sur = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

plt.bar(pclass_sur['Pclass'],pclass_sur['Survived'])

plt.title('Survival Rate Pclass Based')

plt.xlabel('P Class Score')

plt.ylabel('Survival Rate')

plt.grid()

plt.show()
sex_sur = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

plt.bar(sex_sur['Sex'], sex_sur['Survived'])

plt.title('Survival Rate Sex Based')

plt.xlabel('Sex Type')

plt.ylabel('Survival Rate')

plt.grid(axis = 'y')

plt.show()

sibsp_sur = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

plt.bar(sibsp_sur['SibSp'],sibsp_sur['Survived'])

plt.title('Survival Rate SibSp Based')

plt.xlabel('SibSp Value')

plt.ylabel('Survival Rate')

plt.xticks(np.arange(min(sibsp_sur['SibSp']), max(sibsp_sur['SibSp'])+1, 1.0))

plt.grid(axis = 'y')

plt.show()
parch = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

plt.bar(parch['Parch'],parch['Survived'])

plt.title('Survival Rate Parch Based')

plt.xlabel('Parch Value')

plt.ylabel('Survival Rate')

plt.xticks(np.arange(min(parch['Parch']), max(parch['Parch'])+1, 1.0))

plt.grid(axis = 'y')

plt.show()
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=10)

g.set(ylabel='Count')
grid = sns.FacetGrid(train_df, row='Survived', col='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.7, bins=20)

grid.set(ylabel='Count')

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, col='Embarked', row='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.7, ci=None)

grid.set(ylabel='Fare')

grid.add_legend()
print("Before Combining", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket','Cabin'], axis=1)

test_df = test_df.drop(['Ticket','Cabin'], axis=1)

combine = [train_df, test_df]



print("After Combining ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# def set_Cabin_type(df):

#     df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

#     df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

#     return df



# train_df = set_Cabin_type(train_df)

# test_df = set_Cabin_type(test_df)



# train_df = pd.get_dummies(train_df, columns = ['Cabin'])

# test_df = pd.get_dummies(test_df, columns = ['Cabin'])

# combine = [train_df, test_df]

# train_df.head()
for data in combine:

    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for data in combine:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',

                                           'Jonkheer', 'Dona'],

                                           'Rare')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in combine:

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for data in combine:

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, col='Pclass', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.7, bins=20)

grid.set(ylabel='Count')

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for data in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = data[(data['Sex'] == i) & (data['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1), 'Age'] = guess_ages[i,j]



    data['Age'] = data['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for data in combine:    

    data.loc[ data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4

train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for data in combine:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for data in combine:

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for data in combine:

    data['Age*Class'] = data.Age * data.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()
freq_port = train_df.Embarked.dropna().mode()

freq_port = freq_port[0]

freq_port
for data in combine:

    data['Embarked'] = data['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df = pd.get_dummies(train_df, columns = ['Embarked'])

test_df = pd.get_dummies(test_df, columns = ['Embarked'])

combine = [train_df, test_df]

# for dataset in combine:

#     dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for data in combine:

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head()
test_df.corr()
train_df.corr()
test_df.corr()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

print('X_train.shape : ', X_train.shape,

      '\nY_train.shape : ', Y_train.shape,

      '\nX_test.shape  : ', X_test.shape)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report



x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=5, stratify = Y_train)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

# print(classification_report(y_test, y_pred))



logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))





# svc.fit(X_train, Y_train)

# Y_pred = svc.predict(X_test)

# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# acc_svc



# submission = pd.DataFrame({

#         "PassengerId": test_df["PassengerId"],

#         "Survived": Y_pred

#     })

# submission.to_csv('submission.csv', index=False)
# KNN



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, Y_train)

# Y_pred = knn.predict(X_test)

# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

acc_gaussian = round(accuracy_score(y_test, y_pred) * 100, 2)



# gaussian.fit(X_train, Y_train)

# Y_pred = gaussian.predict(X_test)

# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# perceptron.fit(X_train, Y_train)

# Y_pred = perceptron.predict(X_test)

# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# acc_perceptron
#Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# linear_svc.fit(X_train, Y_train)

# Y_pred = linear_svc.predict(X_test)

# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# sgd.fit(X_train, Y_train)

# Y_pred = sgd.predict(X_test)

# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# acc_sgd
#Gradient Boosting

gradient_boost = GradientBoostingClassifier()

gradient_boost.fit(x_train, y_train)

y_pred = gradient_boost.predict(x_test)

acc_gradient_boost = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# gradient_boost.fit(X_train, Y_train)

# Y_pred = gradient_boost.predict(X_test)

# gradient_boost.score(X_train, Y_train)

# acc_gradient_boost = round(gradient_boost.score(X_train, Y_train) * 100, 2)

# acc_gradient_boost
# Bagging Classifier



bagging = BaggingClassifier()

bagging.fit(x_train, y_train)

y_pred = bagging.predict(x_test)

acc_bagging = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# bagging.fit(X_train, Y_train)

# Y_pred = bagging.predict(X_test)

# bagging.score(X_train, Y_train)

# acc_bagging = round(bagging.score(X_train, Y_train) * 100, 2)

# acc_bagging
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)

# print(classification_report(y_test, y_pred))



# decision_tree.fit(X_train, Y_train)

# Y_pred = decision_tree.predict(X_test)

# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# acc_decision_tree
# Random Forest

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report



random_forest = RandomForestClassifier(n_estimators = 117)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

acc_random_forest = round(accuracy_score(y_test, y_pred) * 100, 2)



# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict(X_test)

# random_forest.score(X_train, Y_train)

# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','Bagging Classifier',

             'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree,

              acc_bagging, acc_gradient_boost]})

models.sort_values(by='Score', ascending=False)
# submission = pd.DataFrame({

#         "PassengerId": test_df["PassengerId"],

#         "Survived": Y_pred

#     })

# submission.to_csv('submission.csv', index=False)