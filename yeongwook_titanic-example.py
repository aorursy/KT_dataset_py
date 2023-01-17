# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.shape
train_df.describe()
train_df.describe(include=['O'])
train_df.info()
train_df.isnull().sum()
test_df.head()
test_df.isnull().sum()
survived = train_df[train_df['Survived']==1]
not_survived = train_df[train_df['Survived']==0]
print("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train_df)*100))
print("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train_df)*100.0))
print("Total: %i"%len(train_df))
train_df.Pclass.value_counts()
train_df.groupby('Pclass').Survived.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived',data=train_df)
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(1, 2, figsize = (18,8))
train_df['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train_df, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
f, ax = plt.subplots(1, 2, figsize=(18,8))
train_df['Survived'][train_df['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
train_df['Survived'][train_df['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)

ax[0].set_title('Survived(male)')
ax[1].set_title('Survived(fmale)')

plt.show()

pd.crosstab([train_df['Sex'], train_df['Survived']], train_df['Pclass'], margins=True).style.background_gradient(cmap='summer_r')
tab = pd.crosstab(train_df['Pclass'], train_df['Sex'])
print(tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
sns.factorplot('Sex', 'Survived', hue='Pclass', height = 4, aspect = 2, data=train_df)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train_df)
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x='Embarked', y="Age", hue='Survived', data=train_df, split = True, ax=ax1)
sns.violinplot(x='Pclass', y="Age", hue='Survived', data=train_df, split = True, ax=ax2)
sns.violinplot(x='Sex', y="Age", hue='Survived', data=train_df, split = True, ax=ax3)
total_survived = train_df[train_df['Survived']==1]
total_not_survived = train_df[train_df['Survived']==0]
male_survived = train_df[(train_df['Survived']==1)&(train_df['Sex']=='male')]
female_survived = train_df[(train_df['Survived']==1)&(train_df['Sex']=='female')]
male_not_survived = train_df[(train_df['Survived']==0)&(train_df['Sex']=='male')]
female_not_survived = train_df[(train_df['Survived']==0)&(train_df['Sex']=='female')]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])
plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
plt.figure(figsize=(15,6))
sns.heatmap(train_df.drop('PassengerId', axis=1).corr(), vmax=0.6, square=True, annot=True)
train_test_data = [train_df, test_df]

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss":2, "Mrs":3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
train_df.head()
train_df.isnull().sum()
for dataset in train_test_data:
    dataset['Embarked']= dataset['Embarked'].fillna('S')
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=train_df, ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=train_df, ax=ax[0, 1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=train_df, ax=ax[1, 0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=train_df, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.show()
train_df.isnull().sum()
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
dataset.head()
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg-age_std, age_avg + age_std, size = age_null_count)
    
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
train_df.head()
for dataset in train_test_data:
    dataset.loc[ dataset['Age']<=16, 'Age']=0
    dataset.loc[ (dataset['Age']>16)&(dataset['Age']<=32), 'Age'] = 1
    dataset.loc[ (dataset['Age']>32)&(dataset['Age']<=48), 'Age'] = 2
    dataset.loc[ (dataset['Age']>48)&(dataset['Age']<=64), 'Age'] = 3
    dataset.loc[ dataset['Age']>64, 'Age']=4
train_df.head()
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train_df['Fare'].median())
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print (train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
train_df.head()
for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=7.91, 'Fare']=0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454), 'Fare']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31), 'Fare']=2
    dataset.loc[dataset['Fare']>31, 'Fare']=3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df.head()
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train_df.drop(features_drop, axis = 1)
test = test_df.drop(features_drop, axis = 1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis = 1)
train.head()
test.head()
X_train=train.drop('Survived', axis = 1)
y_train=train['Survived']
X_test = test.drop("PassengerId", axis = 1).copy()

X_train.shape, y_train.shape, X_test.shape
X_test
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X_train, y_train)*100, 2)
print(str(acc_log_reg) + ' percent')
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train)*100, 2)
print (acc_svc)
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train)*100, 2)
print(acc_linear_svc)
clf = KNeighborsClassifier(n_neighbors = 4)
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train)*100, 2)
print(acc_knn)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_decision_tree  = round(clf.score(X_train, y_train)*100, 2)
print(acc_decision_tree )
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron)
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)
from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train)*100, 2)

class_names = ['Survived', 'Not Survived']

cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print('Confusion Matrix in numbers')
print(cnf_matrix)
print(' ')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)




models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)
test.head()
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_random_forest
    })

submission.to_csv('submission_my.csv', index=False)