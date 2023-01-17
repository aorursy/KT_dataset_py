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
# On stock le chemin relatif du dossier contenant nos csv dans une variable

input_path = "../input/titanic/"

# On utilise la fonction pandas pour lire et stocker le contenu du csv dans une variable

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
df_train.info()
df_train.isna().sum()
df_train.describe()
# Création d'une nouvelle colonne pour garder l'information sur les âges null ou non

df_train['age_is_null'] = np.where(df_train['Age'].isnull() , 1, 0)

df_test['age_is_null'] = np.where(df_test['Age'].isnull() , 1, 0)
df_train['Age'].fillna(df_train['Age'].mean(), inplace = True)

df_test['Age'].fillna(df_test['Age'].mean(), inplace = True)
df_train.isna().sum()
df_train
df_train['Embarked'].fillna('S', inplace = True)

df_test['Embarked'].fillna('S', inplace = True)
df_train.isna().sum()
df_train['cabin_is_null'] = np.where(df_train['Cabin'].isnull() , 1, 0)

df_test['cabin_is_null'] = np.where(df_test['Cabin'].isnull() , 1, 0)
df_test.isna().sum()
df_test['fare_is_null'] = np.where(df_test['Fare'].isnull() , 1, 0)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace = True)
df_test.isna().sum()
X_train = df_train.loc[:, ['Pclass', 'SibSp']]
# On stock uniquement la colonne 'Survived' dans une variable y_train

y_train = df_train['Survived']
X_train
X_test = df_test.loc[:, ['Pclass', 'SibSp']]
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_train);
sns.catplot(x="Pclass", y="Age", kind="bar", data=df_train);
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
from sklearn.model_selection import train_test_split

def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))

    test_acc = accuracy_score(y_test, model.predict(X_test))

    return {

        'train accuracy': train_acc,

        'test accuracy': test_acc

    }

print(train_model(rf, X_train, y_train))

# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)