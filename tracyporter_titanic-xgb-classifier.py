#import libraries

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from itertools import combinations

from sklearn import preprocessing as pp

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc, roc_auc_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#read files

#Reading train file:

train = pd.read_csv('/kaggle/input/titanic/train.csv')

#Reading test file:

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train
train.info()
train.describe()
test
test.info()
test.describe()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(4)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(4)
# fill up missing values with mode

train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])

test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])



train['Fare'] = train['Fare'].fillna(train['Fare'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])



train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])



#fill missing values with median

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
train.isnull().sum().sum(), test.isnull().sum().sum()
train
test
#take title from name

train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
title=train.groupby('Title')['Survived'].sum().reset_index()

title
title1={'Capt':1, 'Col':2, 'Don':3, 'Dr':4,'Jonkheer':5, 'Lady':6, 'Major': 7, 'Master':8, 'Miss':9, 

        'Mlle':10, 'Mme':11, 'Mr':12, 'Mrs':13, 'Ms':14, 'Rev':15, 'Sir':16, 'the Countess':17, 'Dona':18}

train.Title=train.Title.map(title1)

test.Title=test.Title.map(title1)
title2=train.groupby('Title')['Survived'].sum().reset_index()

title2
train['Title'].isnull().sum().sum(), test['Title'].isnull().sum().sum()
pclass=train.groupby('Pclass')['Survived'].sum().reset_index()

pclass
pclass = train.Pclass.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

sns.barplot(x=pclass.index, y=pclass.values)

plt.show()
Class=train.groupby('Pclass')['Survived'].sum().reset_index()

Class
train['Pclass'].isnull().sum().sum(), test['Pclass'].isnull().sum().sum()
sex=train.groupby('Sex')['Survived'].sum().reset_index()

sex
sex = train.Sex.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

sns.barplot(x=sex.index, y=sex.values)

plt.show()
sex1={'female':1, 'male':0}

train.Sex=train.Sex.map(sex1)

test.Sex=test.Sex.map(sex1)
sex2=train.groupby('Sex')['Survived'].sum().reset_index()

sex2
age=train.groupby('Age')['Survived'].sum().reset_index()

age
plt.figure(figsize=(10,6))

plt.title("Ages Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Age"])

plt.show()
age18_25 = train.Age[(train.Age <= 25) & (train.Age >= 18)]

age26_35 = train.Age[(train.Age <= 35) & (train.Age >= 26)]

age36_45 = train.Age[(train.Age <= 45) & (train.Age >= 36)]

age46_55 = train.Age[(train.Age <= 55) & (train.Age >= 46)]

age55above = train.Age[train.Age >= 56]



x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=x, y=y, palette="rocket")

plt.title("Number of Survivors and Ages")

plt.xlabel("Age")

plt.ylabel("Number of Survivors")

plt.show()
bins = [0., 18., 35., 64., 65.+ np.inf]

names = ['child','young adult', 'middle aged', 'pensioner']



train['Age_Range'] = pd.cut(train['Age'], bins, labels=names)

test['Age_Range'] = pd.cut(test['Age'], bins, labels=names)
age_range=train.groupby('Age_Range')['Survived'].sum().reset_index()

age_range
age_range1={'child':1,'young adult':2, 'middle aged':3, 'pensioner': 4}

train.Age_Range=train.Age_Range.map(age_range1)

test.Age_Range=test.Age_Range.map(age_range1)
train.Age_Range.isnull().sum(), test.Age_Range.isnull().sum()
age_range=train.groupby('Age_Range')['Survived'].sum().reset_index()

age_range
family=train.groupby('SibSp')['Survived'].sum().reset_index()

family
plt.figure(figsize=(10,6))

plt.title("Family Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["SibSp"])

plt.show()
parch=train.groupby('Parch')['Survived'].sum().reset_index()

parch
plt.figure(figsize=(10,6))

plt.title("Parch Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Parch"])

plt.show()
fare=train.groupby('Fare')['Survived'].sum().reset_index()

fare
plt.figure(figsize=(10,6))

plt.title("Fare Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Fare"])

plt.show()
bins0 = [-1., 100., 200., 300., 400., 500.+ np.inf]

names0 = ['0-99', '100-199', '200-299', '300-399', '400+']



train['Fare_Range'] = pd.cut(train['Fare'], bins0, labels=names0)

test['Fare_Range'] = pd.cut(test['Fare'], bins0, labels=names0)
fare_range=train.groupby('Fare_Range')['Survived'].sum().reset_index()

fare_range
fare_range1={'0-99':1, '100-199':2, '200-299': 3, '300-399':0, '400+':5}

train.Fare_Range=train.Fare_Range.map(fare_range1)

test.Fare_Range=test.Fare_Range.map(fare_range1)
train.Fare_Range.isnull().sum(), test.Fare_Range.isnull().sum()
train['Fare_Range'] = train['Fare_Range'].fillna(train['Fare_Range'].mode()[0])

test['Fare_Range'] = test['Fare_Range'].fillna(test['Fare_Range'].mode()[0])
embark=train.groupby('Embarked')['Survived'].sum().reset_index()

embark
embark1={'C':1, 'Q':2, 'S': 3}

train.Embarked=train.Embarked.map(embark1)

test.Embarked=test.Embarked.map(embark1)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(3)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(3)
train.isnull().sum().sum(), test.isnull().sum().sum()
train
test
train.dtypes
test.dtypes
#create a heatmap to correlate survival

plt.figure(figsize=(6,4))

cmap=train.corr()

sns.heatmap(cmap, annot=True)

y = train["Survived"]

features = ["Pclass", "Sex", "Parch", "Embarked", "Title", 

            "Age_Range", "SibSp", "Fare_Range"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
X
X_test
#split train set for testing

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#bring all features to the same range

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_validation=sc_X.transform(X_validation)
model= XGBClassifier()

name='XGB'
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Make predictions on validation dataset

model = XGBClassifier(learning_rate=1, n_estimators=2000, max_depth=40, min_child_weight=40, 

                      gamma=0.4,nthread=10, subsample=0.8, colsample_bytree=.8, 

                      objective= 'binary:logistic',scale_pos_weight=10,seed=29)

model.fit(X, y)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, model.predict(X))

print(auc(false_positive_rate, true_positive_rate))
print(roc_auc_score(y, model.predict(X)))
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#upload submission

my_submission = pd.read_csv("my_submission.csv")

my_submission