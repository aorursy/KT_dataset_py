import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.isnull().sum()
test.isnull().sum()
%matplotlib inline

ax = train['Pclass'].value_counts().plot(kind='bar')
ax = train['Survived'].value_counts().plot(kind = 'bar')
plt.hist(train['Age'],range = (0,80))
train["Embarked"].value_counts()
sex_dummies_train  = pd.get_dummies(train['Sex'])

sex_dummies_train.columns = ['male','female']



sex_dummies_test  = pd.get_dummies(test['Sex'])

sex_dummies_test.columns = ['male','female']



train.drop(['Sex'],axis=1,inplace=True)

test.drop(['Sex'],axis=1,inplace=True)



train= train.join(sex_dummies_train)

test= test.join(sex_dummies_test)
train['SibSp'].value_counts()

SS_dummies_train  = pd.get_dummies(train['SibSp'])

SS_dummies_train.columns = ['c1','c2','c3','c4','c5','c6','c7']



SS_dummies_test  = pd.get_dummies(test['SibSp'])

SS_dummies_test.columns = ['c1','c2','c3','c4','c5','c6','c7']



train.drop(['SibSp'],axis=1,inplace=True)

test.drop(['SibSp'],axis=1,inplace=True)



train= train.join(SS_dummies_train)

test= test.join(SS_dummies_test)
train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"].fillna(test["Fare"].median(), inplace=True)
train_age_mean  = train["Age"].mean()

train_age_std      = train["Age"].std()

train_age_null = train["Age"].isnull().sum()



test_age_mean   = test["Age"].mean()

test_age_std       = test["Age"].std()

test_age_null = test["Age"].isnull().sum()



s1 = pd.Series(np.random.randint(train_age_mean - train_age_std , train_age_mean + train_age_std , size = train_age_null))

s2 = pd.Series(np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std , size = test_age_null))



train["Age"]= train["Age"].fillna(np.random.randint(train_age_mean - train_age_std , train_age_mean + train_age_std) )

test["Age"]= test["Age"].fillna(np.random.randint(test_age_mean - test_age_std , test_age_mean + test_age_std) )

train['Age'].isnull().sum()

test["Age"].isnull().sum()
train['Parch'].value_counts()
test.drop("Cabin",axis=1,inplace=True)

train.drop("Cabin",axis=1,inplace=True)
test.isnull().sum()
train.isnull().sum()
pclass_dummies_train  = pd.get_dummies(train['Pclass'])

pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train.drop(['Pclass'],axis=1,inplace=True)

test.drop(['Pclass'],axis=1,inplace=True)



train= train.join(pclass_dummies_train)

test= test.join(pclass_dummies_test)
train.corr()
X_train = train[['Age', 'Fare', 'male',

       'female', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'Class_1',

       'Class_2']]

Y_train = train["Survived"]

X_test  = test[['Age', 'Fare', 'male',

       'female', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'Class_1',

       'Class_2']]
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import AdaBoostClassifier

X_train.columns
predictors =['Age', 'Fare', 'male',

       'female', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'Class_1',

       'Class_2']

alg = AdaBoostClassifier()

select = SelectKBest(f_classif, k=5)

select.fit(X_train[predictors], Y_train)
scores = -np.log10(select.pvalues_)
alg.fit(X_train,Y_train)
predictions = alg.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": predictions})
submission.to_csv('prediction.csv',index= False)