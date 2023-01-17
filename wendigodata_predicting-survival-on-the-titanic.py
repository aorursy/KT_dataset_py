import pandas as pd

import numpy as np

import csv



import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('train.csv', header=0)

df_test = pd.read_csv('test.csv', header=0)
df.loc[df.PortCode.isnull(), "PortCode"] = 1
df['FamilySize'] = df["Parch"]+df['SibSp']

df['Agefill*Pclass'] = df.Agefill * df.Pclass

titles = ["Dr.", "Don.", "General", "Colonel", "Captain", "Count", "Countess", "Jonkheer", "Lord", "Master.", "Sir.", "Mlle.",

         "Cpt.", "Major.", "Col.", "Mme", "Lady."]

df['Title'] = 0

for title in titles:

    df.loc[(df['Name'].str.contains(title, regex=False)), 'Title'] = 1
df_test['Gender'] = df_test['Sex'].map(lambda x: x[0].upper())

df_test['Gender'] = df_test['Sex'].map( {'female':0, 'male':1} ).astype(int)



df_test['PortCode'] = df_test['Embarked'].dropna()

df_test['PortCode'] = df_test['Embarked'].dropna().map( {'C':0, 'S':1, 'Q':2}).astype(float)

df_test['Agefill'] = df_test["Age"]

for i in range(0, 2):

    for j in range(0,3):

        for k in range(0,3):

            median_ages[i,j,k] = df_test[(df_test['Gender'] == i) & (df_test['Pclass'] == j+1) & (df_test['PortCode'] == k)]['Age'].dropna().median()



for i in range(0, 2):

    for j in range(0,3):

        for k in range(0,3):

            df_test.loc[ (df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1) & (df_test.PortCode == k), 'Agefill'] = median_ages[i, j, k]
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.4, random_state=0)

test, validate = train_test_split(test, test_size=0.5, random_state=0)
train_data = train.values

test_data = test.values

validate_data = validate.values