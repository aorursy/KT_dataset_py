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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
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
pclass = train.Pclass.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=pclass.index, y=pclass.values)
plt.show()
bins4 = [-1., 1., 2., 3. + np.inf]
names4 = ['1','2', '3']

train['Class_Range'] = pd.cut(train['Pclass'], bins4, labels=names4)
test['Class_Range'] = pd.cut(test['Pclass'], bins4, labels=names4)
class_range=train.groupby('Class_Range')['Survived'].sum().reset_index()
class_range
train['Class_Range'].isnull().sum().sum(), test['Class_Range'].isnull().sum().sum()
sex=train.groupby('Sex')['Survived'].sum().reset_index()
sex
sex1={'male':0, 'female':1}
train.Sex=train.Sex.map(sex1)
test.Sex=test.Sex.map(sex1)
sex = train.Sex.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=sex.index, y=sex.values)
plt.show()
bins6 = [-1., 0, 1. +np.inf]
names6 = ['0','1']

train['Sex_Range'] = pd.cut(train['Sex'], bins6, labels=names6)
test['Sex_Range'] = pd.cut(test['Sex'], bins6, labels=names6)
sex1=train.groupby('Sex_Range')['Survived'].sum().reset_index()
sex1
train.Sex_Range.isnull().sum(), test.Sex_Range.isnull().sum()
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
bins2 = [-1., 0., 1., 2., 3., 4., 5., 8.+ np.inf]
names2 = ['0','1', '2', '3', '4', '5', '8']

train['Family_Range'] = pd.cut(train['SibSp'], bins2, labels=names2)
test['Family_Range'] = pd.cut(test['SibSp'], bins2, labels=names2)
family1=train.groupby('Family_Range')['Survived'].sum().reset_index()
family1
parch=train.groupby('Parch')['Survived'].sum().reset_index()
parch
plt.figure(figsize=(10,6))
plt.title("Parch Frequency")
sns.axes_style("dark")
sns.violinplot(y=train["Parch"])
plt.show()
bins3 = [-1., 0., 1., 2., 3., 4., 5., 6.+ np.inf]
names3 = ['0','1', '2', '3', '4', '5', '6']

train['Parch_Range'] = pd.cut(train['SibSp'], bins3, labels=names3)
test['Parch_Range'] = pd.cut(test['SibSp'], bins3, labels=names3)
parch1=train.groupby('Parch_Range')['Survived'].sum().reset_index()
parch1
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
bins1 = [-1., 1., 2., 3.+ np.inf]
names1 = ['C', 'Q', 'S']

train['Embarked_Range'] = pd.cut(train['Embarked'], bins1, labels=names1)
test['Embarked_Range'] = pd.cut(test['Embarked'], bins1, labels=names1)
train.Embarked_Range.isnull().sum(), test.Embarked_Range.isnull().sum()
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
drop_elements = ['Name', 'Ticket', 'Cabin', 'Title']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
y = train["Survived"]
features = ["Class_Range", "Sex_Range", "Embarked_Range", "Age_Range", "Family_Range", "Parch_Range", "Fare_Range"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
X
X_test
#split train set for training and testing
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, stratify=y, test_size=0.25, random_state=101)
# distribution in training set
Y_train.value_counts(normalize=True)
# distribution in validation set
Y_validation.value_counts(normalize=True)
#shape of training set
X_train.shape, Y_train.shape
#shape of validation set
X_validation.shape, Y_validation.shape
#creating the decision tree function
dt_model = DecisionTreeClassifier(random_state=10)
#fitting the model
dt_model.fit(X_train, Y_train)
#checking the training score
dt_model.score(X_train, Y_train)
#checking the validation score
dt_model.score(X_validation, Y_validation)
#predictions on validation set
dt_model.predict(X_validation)
dt_model.predict_proba(X_validation)
y_pred = dt_model.predict_proba(X_validation)[:,1]
y_pred
y_new = []
for i in range(len(y_pred)):
    if y_pred[i]<=0.7:
        y_new.append(0)
    else:
        y_new.append(1)
y_new
accuracy_score(Y_validation, y_new)
#predictions on test set
dt_model.predict(X_test)
y_pred1 = dt_model.predict_proba(X_test)
y_pred1 = dt_model.predict_proba(X_test)[:,1]
y_pred1
y_new1 = []
for i in range(len(X_test)):
    if y_pred1[i]<=0.7:
        y_new1.append(0)
    else:
        y_new1.append(1)
y_new1
submission = pd.DataFrame({
        "PassengerId": test.PassengerId,
        "Survived": y_new1
    })
submission.to_csv('submission.csv', index=False)
submission