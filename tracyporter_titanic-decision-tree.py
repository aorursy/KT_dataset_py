#algorithm

#https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset
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

from sklearn.model_selection import KFold

from sklearn import tree
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

# Feature selection: remove variables no longer containing relevant information

drop_elements = ['Name', 'Ticket', 'Cabin']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)
y = train["Survived"]

features = ["Pclass", "Sex", "Embarked", "Title", 

            "Age_Range", "SibSp", "Parch", "Fare_Range"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
X
X_test
#split train set for testing

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, stratify=y, test_size=0.20, random_state=101)
cv = KFold(n_splits=10)            # Desired number of Cross Validation folds

accuracies = list()

max_attributes = len(list(test))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)

    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train):

        f_train = train.loc[train_fold] # Extract train data with cv indices

        f_valid = train.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    # print("Accuracy per fold: ", fold_accuracy, "\n")

    # print("Average accuracy: ", avg)

    # print("\n")

    

# Just to show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models

y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values



# Create Decision Tree with max_depth = 3

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)

decision_tree.fit(x_train, y_train)



# Predicting results for test dataset

y_pred = decision_tree.predict(x_test)

submission = pd.DataFrame({

        "PassengerId": test.PassengerId,

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)

submission
