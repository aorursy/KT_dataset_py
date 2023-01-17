from tpot import TPOTClassifier

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd
titanic = pd.read_csv('../input/titanic/train.csv')

titanic.head(20)
titanic.groupby('Sex').Survived.value_counts()
titanic.groupby(['Pclass','Sex']).Survived.value_counts()
id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(float))

id.div(id.sum(1).astype(float), 0)
titanic.rename(columns={'Survived': 'class'}, inplace=True)
titanic.dtypes
for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:

    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, titanic[cat].unique().size))
for cat in ['Sex', 'Embarked']:

    print("Levels for catgeory '{0}': {1}".format(cat, titanic[cat].unique()))
titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})

titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})
titanic = titanic.fillna(-999)

pd.isnull(titanic).any()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])
CabinTrans
titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)
assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done
titanic_new = np.hstack((titanic_new.values,CabinTrans))
titanic_new[0].size
titanic_class = titanic['class'].values
training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)

training_indices.size, validation_indices.size
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)

tpot.fit(titanic_new[training_indices], titanic_class[training_indices])
tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)
titanic1 = pd.read_csv('../input/titanic/test.csv')
titanic1.describe()
for var in ['Cabin']: #,'Name','Ticket']:

    new = list(set(titanic1[var]) - set(titanic1[var]))

    titanic1.ix[titanic1[var].isin(new), var] = -999
titanic1['Sex'] = titanic1['Sex'].map({'male':0,'female':1})

titanic1['Embarked'] = titanic1['Embarked'].map({'S':0,'C':1,'Q':2})
titanic1 = titanic1.fillna(-999)

pd.isnull(titanic1).any()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic1['Cabin'].values])

titanic1 = titanic1.drop(['Name','Ticket','Cabin'], axis=1)
titanic1_new = np.hstack((titanic1.values,SubCabinTrans))
np.any(np.isnan(titanic1_new))
assert (titanic1_new.shape[1] == titanic1_new.shape[1]), "Not Equal"
submission = tpot.predict(titanic1_new)
final = pd.DataFrame({'PassengerId': titanic1['PassengerId'], 'Survived': submission})
final.to_csv('MySubmission.csv', index=False)