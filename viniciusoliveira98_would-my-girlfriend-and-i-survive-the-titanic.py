import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))

display(train.dtypes)

display(train.head())
train.describe()
((train.isnull().sum() / train.shape[0]).sort_values(ascending=False))*100 
train.hist(figsize=(10,8))

(train[['Sex','Survived']].groupby(['Sex']).mean())*100
sns.countplot(x='Sex', data=train)
sns.countplot(x='Pclass', data=train)
sns.countplot(x='Embarked', data=train)
age_survived = sns.FacetGrid(train, col='Survived')
age_survived.map(sns.distplot, 'Age')
train.describe(include=['O'])
train_idx = train.shape[0]
test_idx = test.shape[0]
target = train.Survived.copy()
train.drop(['Survived'], axis=1, inplace=True)

df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_merged.isnull().sum()
age_mean = df_merged['Age'].mean()
df_merged['Age'].fillna(age_mean, inplace=True)
fare_top = df_merged['Fare'].value_counts()[0]
df_merged['Fare'].fillna(fare_top, inplace=True)
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)
display(df_merged.head())
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(train, target)
acc_logReg = round(lr_model.score(train, target) * 100, 2)
print("Accuracy of the model Logistic Regression: {}".format(acc_logReg))
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(train, target)
acc_tree = round(tree_model.score(train, target) * 100, 2)
print("Accuracy of the model Decision tree: {}".format(acc_tree))
print("Would my girlfriend and I survive the titanic? ")
vinicius_oliveira = np.array([3,0,22,1,0,15.5,0,0,0,1]).reshape((1,-1))
giovanna_lyssa = np.array([3,1,20,1,0,15.5,0,0,0,1]).reshape((1,-1))
print("Vinícius:\t{}".format(tree_model.predict(vinicius_oliveira)[0]))
print("Giovanna:\t{}".format(tree_model.predict(giovanna_lyssa)[0]))
