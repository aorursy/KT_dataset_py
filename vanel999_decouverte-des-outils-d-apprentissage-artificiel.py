import numpy as np 
import pandas as pd 
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
# Charger les données
df_trainingSet = pd.read_csv('../input/train.csv')
df_testingSet  = pd.read_csv('../input/test.csv')

combine = [df_trainingSet, df_testingSet]
df_dataset = pd.concat(combine)
df_trainingSet.head()
df_trainingSet.tail()
df_trainingSet.info()
df_testingSet.info()
df_trainingSet.describe()
df_trainingSet.describe(include=['O'])
df_trainingSet[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_trainingSet[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_trainingSet[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_trainingSet[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_sex = pd.DataFrame([df_trainingSet[df_trainingSet.Survived == 1]['Embarked'].value_counts(), 
                       df_trainingSet[df_trainingSet.Survived == 0]['Embarked'].value_counts()])
df_sex.index = ['Survived','Died']

display(df_sex)
g = sns.FacetGrid(df_trainingSet, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_trainingSet, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

df_aux = df_trainingSet[ ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' ,'Embarked'] ]
freq_port = df_trainingSet.Embarked.dropna().mode()[0]
freq_port
df_trainingSet['Embarked'] = df_trainingSet['Embarked'].fillna(freq_port)


df_trainingSet[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Les attributs ayant des valeurs manquantes dans le testingSet (Fare et Age)
df_trainingSet['Age'].fillna(df_trainingSet['Age'].dropna().median(), inplace=True)
df_trainingSet['Sex'] = df_trainingSet['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
df_trainingSet['Embarked'] = df_trainingSet['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_trainingSet['AgeBand'] = pd.cut(df_trainingSet['Age'], 5)
df_trainingSet[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
df_trainingSet.loc[ df_trainingSet['Age'] <= 16, 'Age'] = 0
df_trainingSet.loc[(df_trainingSet['Age'] > 16) & (df_trainingSet['Age'] <= 32), 'Age'] = 1
df_trainingSet.loc[(df_trainingSet['Age'] > 32) & (df_trainingSet['Age'] <= 48), 'Age'] = 2
df_trainingSet.loc[(df_trainingSet['Age'] > 48) & (df_trainingSet['Age'] <= 64), 'Age'] = 3
df_trainingSet.loc[df_trainingSet['Age'] > 64, 'Age']

df_trainingSet = df_trainingSet.drop(['AgeBand'], axis=1)
df_trainingSet['FamilySize'] = df_trainingSet['SibSp'] + df_trainingSet['Parch'] + 1
df_trainingSet[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_trainingSet['IsAlone'] = 0
df_trainingSet.loc[df_trainingSet['FamilySize'] == 1, 'IsAlone'] = 1
df_trainingSet[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df_trainingSet = df_trainingSet.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_trainingSet.head()
df_trainingSet['FareBand'] = pd.qcut(df_trainingSet['Fare'], 4)
df_trainingSet[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
df_trainingSet.loc[ df_trainingSet['Fare'] <= 7.91, 'Fare'] = 0
df_trainingSet.loc[(df_trainingSet['Fare'] > 7.91) & (df_trainingSet['Fare'] <= 14.454), 'Fare'] = 1
df_trainingSet.loc[(df_trainingSet['Fare'] > 14.454) & (df_trainingSet['Fare'] <= 31), 'Fare']   = 2
df_trainingSet.loc[ df_trainingSet['Fare'] > 31, 'Fare'] = 3
df_trainingSet['Fare'] = df_trainingSet['Fare'].astype(int)

df_trainingSet = df_trainingSet.drop(['FareBand'], axis=1)
df_trainingSet.head()
df_titanic_tr = df_trainingSet.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_titanic_tr.head()
df_tr_inputData = df_titanic_tr.drop("Survived", axis=1)
df_tr_class = df_titanic_tr["Survived"]
decision_tree = DecisionTreeClassifier()
decision_tree.fit(df_tr_inputData, df_tr_class)
acc_decision_tree = round(decision_tree.score(df_tr_inputData, df_tr_class) * 100, 2)
acc_decision_tree
gaussian = GaussianNB()
gaussian.fit(df_tr_inputData, df_tr_class)
acc_gaussian = round(gaussian.score(df_tr_inputData, df_tr_class) * 100, 2)
acc_gaussian
perceptron = Perceptron()
perceptron.fit(df_tr_inputData, df_tr_class)
acc_perceptron = round(perceptron.score(df_tr_inputData, df_tr_class) * 100, 2)
acc_perceptron
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(df_tr_inputData, df_tr_class)
acc_knn = round(knn.score(df_tr_inputData, df_tr_class) * 100, 2)
acc_knn
for K in range(25):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
 neigh.fit(df_tr_inputData, df_tr_class) 
 acc_knn = round(neigh.score(df_tr_inputData, df_tr_class) * 100, 2)
 print ("Accuracy  =  ", acc_knn ,"% pour la k = :",K_value)