# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train,test]
print(train.info())

print('_'*40)

print(test.info())
train.describe()
train.describe(include=['O'])
round(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending = False),2)
round(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending = False),2)
round(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending = False),2)
round(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending = False),2)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending = False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending = False)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist,'Age', bins=40)
g = sns.FacetGrid(train, col='Pclass', hue='Survived')

#g = sns.FacetGrid(train, col='Survived', row='Pclass')

g.map(plt.hist, 'Age', alpha=.5,bins=20)

g.add_legend()
g = sns.FacetGrid(train, col='Survived', hue='Pclass')

g.map(plt.hist, 'Age', alpha=.5,bins=20)

g.add_legend()
g = sns.FacetGrid(train, col='Embarked')

#g = sns.FacetGrid(train, row='Embarked', size=2.5,aspect=2)

g.map(sns.pointplot, 'Pclass','Survived','Sex',palette='deep')

g.add_legend()
g = sns.FacetGrid(train, col='Embarked')

g.map(sns.pointplot, 'Sex','Survived','Pclass',palette='deep')

g.add_legend()
g = sns.FacetGrid(train, col='Embarked', hue='Survived', palette={0: 'yellow', 1: 'red'})

#g = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

g.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

g.add_legend()
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
pd.crosstab(test['Title'], test['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Jonkheer','Lady',

                                          'Major','Rev','Dr','Sir','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    

round(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean(),2)
title_mapping = {'Mrs':1, 'Miss':2, 'Master':3, 'Mr':4, 'Rare':5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title']
train.info()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0})



train.head()
train_df = train.drop(['Name', 'PassengerId'], axis=1)

test_df = test.drop(['Name', 'PassengerId'], axis=1)

combine_df = [train_df,test_df]
g = sns.FacetGrid(train_df, row='Pclass',col='Sex', size=3, aspect=1.6)

g.map(plt.hist, 'Age', bins=20)
age_guess = np.zeros((2,3))



for dataset in combine_df:

    for i in range(0, 2):

        for j in range(0, 3):

            guess = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            guess_age = guess.median()



            # Convert random age float to nearest .5 age

            age_guess[i,j] = int( guess_age/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = age_guess[i,j]



    dataset['Age'] = dataset['Age'].astype(int)
freq_emb = train_df.Embarked.dropna().mode()[0]

for dataset in combine_df:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_emb)
train_df[['Survived','Embarked']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine_df:

    dataset.Embarked=dataset.Embarked.map({'C':1, 'Q':2, 'S':3}).astype(int)



train_df.head()
train_df.info()

print("-"*40)

test_df.info()
test_df.Fare.fillna(test_df.Fare.dropna().median(),inplace=True)

test_df.info()
train_df = train_df.drop(['Ticket','Cabin'],axis=1)

test_df = test_df.drop(['Ticket','Cabin'],axis=1)
# Split train data into validation and training data

from sklearn.model_selection import train_test_split



X_train_all = train_df.drop('Survived', axis=1)

Y_train_all = train_df.Survived

X_test = test_df.copy()





X_train, X_val, Y_train, Y_val = train_test_split(X_train_all, Y_train_all,random_state=1, test_size=0.2)

X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape
# Logistic Regression

from sklearn.metrics import f1_score



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_val)

log_score = logreg.score(X_val,Y_val)

log_score, f1_score(Y_val, Y_pred, average='binary') 
coef = pd.DataFrame(train_df.columns.delete(0))

coef.columns = ['Feature']

coef['Coefficient'] = pd.Series(logreg.coef_[0])

coef.sort_values(by='Coefficient', ascending=False)
# Support Vector Machine



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_val)

svc_score = svc.score(X_val,Y_val)

svc_score, f1_score(Y_val, Y_pred, average='binary') 
# k-Nearest Neighbors



knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_val)

knn_score = knn.score(X_train, Y_train)

knn_score, f1_score(Y_val, Y_pred, average='binary')
# Gaussian Naive Bayes



gnb = GaussianNB()

gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_val)

gnb_score = gnb.score(X_val, Y_val)

gnb_score, f1_score(Y_val, Y_pred, average='binary')
# Perceptron



perc = Perceptron()

perc.fit(X_train,Y_train)

Y_pred = perc.predict(X_val)

perc_score = perc.score(X_val, Y_val)

perc_score, f1_score(Y_val, Y_pred, average='binary')
# Linear SVC



lsvc = LinearSVC()

lsvc.fit(X_train,Y_train)

Y_pred = lsvc.predict(X_val)

lsvc_score = lsvc.score(X_val, Y_val)

lsvc_score, f1_score(Y_val, Y_pred, average='binary')
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train,Y_train)

Y_pred = sgd.predict(X_val)

sgd_score = sgd.score(X_val, Y_val)

sgd_score, f1_score(Y_val, Y_pred, average='binary')
# Decision Tree



dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

Y_pred = dt.predict(X_val)

dt_score = round(dt.score(X_val, Y_val),4)

dt_score, f1_score(Y_val, Y_pred, average='binary')
# Random Forest



rf = RandomForestClassifier()

rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_val)

rf_score = round(rf.score(X_val, Y_val),4)

rf_score, f1_score(Y_val, Y_pred, average='binary')
# Random Forest n_estimators=100



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_val)

rf_score = round(rf.score(X_val, Y_val),4)

rf_score, f1_score(Y_val, Y_pred, average='binary')
# #XGBoost



from xgboost import XGBClassifier

xgbr = XGBClassifier(n_estimators=1000, learn_rate=0.05, random_state=1)

xgbr.fit(X_train,Y_train)

Y_pred = xgbr.predict(X_val)

xgbr_score = round(xgbr.score(X_val, Y_val),4)

xgbr_score, f1_score(Y_val, Y_pred, average='binary')
model = LogisticRegression()

model.fit(X_train_all,Y_train_all)

pred = model.predict(X_test)
output = pd.DataFrame({

    'PassengerId': test['PassengerId'],

    'Survived': pred

})



output.to_csv('submission1.1.csv', index=False)