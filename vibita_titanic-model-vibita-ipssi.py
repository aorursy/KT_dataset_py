# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
df_train
df_train.describe()
df_train.info()
df_train.columns
df_test
df_test.info()
df_test.describe()
df_train.drop(['Name','Fare', 'Cabin', 'Ticket'], axis=1)

#Les colonnes ci-dessus ne seront pas utilisé pour se concentrer majoritairement sur les données numériques
df_test.drop(['Name','Fare', 'Cabin', 'Ticket'], axis=1)
for c in df_train.select_dtypes('object').columns:

    if (c == 'Sex') | (c == 'Embarked'):

        continue

    df_train[c] = df_train[c].astype('category').cat.codes
for c in df_test.select_dtypes('object').columns:

    if (c == 'Sex') | (c == 'Embarked'):

        continue

    df_test[c] = df_test[c].astype('category').cat.codes
df_train.isnull().sum()
df_train.isna().sum()/len(df_train)*100

#20% du champs age n'est pas renseigné
#Imputation des données manquantes

#df_train['age_is_null'] = np.where(df_train['Age'].isnull() , 1, 0)



df_train['Age'] = df_train['Age'].replace(0, np.NaN)

df_train.fillna(df_train.mean(), inplace=True)

print(df_train.isnull().sum())
df_test['Age'] = df_test['Age'].replace(0, np.NaN)

df_test.fillna(df_test.mean(), inplace=True)

print(df_test.isnull().sum())
#df_train.loc[df_train['Survived']=='undefined']
X_train = df_train.loc[:, ['SibSp','Parch']]

X_train
y_train = df_train['Survived']

y_train
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);

#Nous pouvons observer qu'environ 70% des survivants sont des femmes
survivedW = df_train.loc[df_train.Sex == 'female']["Survived"]

percentW = sum(survivedW)/len(survivedW)

percentW

#74% de femme ont survécus
survivedM = df_train.loc[df_train.Sex == 'male']["Survived"]

percentM = sum(survivedM)/len(survivedM)

percentM

#Contre seulement 18% d'hommes qui ont survécus
sns.barplot(x='Pclass',y='Survived', data=df_train)

#Nous pouvons observer que la majorité des survivant sont ceux qui étaient en première class dans le bateau
#Age de ceux qui ont survécu

sns.relplot(x='Age',y='Survived', data=df_train)
sns.catplot(x="SibSp", y="Survived", kind="bar", data=df_train);

#Personne accompagné de leur époux(se) ou frère/soeur ont eu moins de chance de survie
sns.catplot(x="Parch", y="Survived", kind="bar", data=df_train);

#Nous pouvons observons qu'un parent avec plusieurs enfants a beaucoup moins de chance de survie que des parents avec moins d'enfants mais ce n'est pas un caractéristique décisif.
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators = 100)
#X = df_train.loc[:, ['SibSp','Parch']]

#y = df_train['Survived']
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
X_test = df_test.loc[:,['SibSp','Parch']]

X_test
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)