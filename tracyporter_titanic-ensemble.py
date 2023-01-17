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

from itertools import combinations

from sklearn import preprocessing as pp

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
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
train['Pclass'].isnull().sum().sum(), test['Pclass'].isnull().sum().sum()
sex=train.groupby('Sex')['Survived'].sum().reset_index()

sex
sex1={'male':0, 'female':1}

train.Sex=train.Sex.map(sex1)

test.Sex=test.Sex.map(sex1)
sex2=train.groupby('Sex')['Survived'].sum().reset_index()

sex2
train.Sex.isnull().sum(), test.Sex.isnull().sum()
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
embark=train.groupby('Embarked')['Survived'].sum().reset_index()

embark
embark1={'C':1, 'Q':2, 'S':3}

train.Embarked=train.Embarked.map(embark1)

test.Embarked=test.Embarked.map(embark1)
train.Embarked.isnull().sum(), test.Embarked.isnull().sum()
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

train.drop('Name',axis=1, inplace=True)

test.drop('Name',axis=1, inplace=True)



train.drop('Cabin',axis=1, inplace=True)

test.drop('Cabin',axis=1, inplace=True)



train.drop('Ticket',axis=1, inplace=True)

test.drop('Ticket',axis=1, inplace=True)
y = train["Survived"]

features = ["Pclass", "Sex", "Parch", "Embarked", "Title", "Age_Range", "SibSp", "Fare_Range"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
X
X_test
from sklearn.impute import SimpleImputer

# Convert the DataFrame object into NumPy array otherwise you will not be able to impute

values = train.values

# Now impute it

imputer = SimpleImputer()

imputedData = imputer.fit_transform(values)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)
#split train set for testing

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, stratify=y, test_size=0.20, random_state=101)
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models

y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values
from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=7)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)

results = model_selection.cross_val_score(model, X, y, cv=kfold)

print(results.mean())
# AdaBoost Classification



from sklearn.ensemble import AdaBoostClassifier

seed = 7

num_trees = 70

kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, X, y, cv=kfold)

print(results.mean())
# Voting Ensemble for Classification



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



kfold = model_selection.KFold(n_splits=10, random_state=seed)

# create the sub models

estimators = []

model1 = LogisticRegression()

estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()

estimators.append(('cart', model2))

model3 = SVC()

estimators.append(('svm', model3))

# create the ensemble model

ensemble = VotingClassifier(estimators)

results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)

print(results.mean())
ensemble
ensemble.fit(x_train, y_train)
# Predicting results for test dataset

y_pred = ensemble.predict(x_test)

submission = pd.DataFrame({

        "PassengerId": test.PassengerId,

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)

submission