import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

import re

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



# Adding a new column so I can split them again

df_train['type'] = "train"

df_test['type'] = "test"



df_full = df_train.append(df_test, sort=False)
df_full.head()
len(df_full)
df_full.describe()
df_full.dtypes
df_full.isnull().sum().plot(kind='bar')
# Split the name to collect the title

df_full['Title'] = df_full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



# Set up a minimum occurrence

min_occ = 10



title_names = (df_full['Title'].value_counts() < min_occ)

df_full['Title'] = df_full['Title'].apply(lambda x: 'Other' if title_names.loc[x] == True else x)

df_full['Title'].value_counts()
# count family size

df_full['f_size'] = df_full['SibSp'] + df_full['Parch'] + 1

df_full['IsAlone'] = 1



# check if the person is alone or not

df_full['IsAlone'].loc[df_full['f_size'] > 1] = 0
# Fill missing Age with mean

df_full['Age'].fillna(df_full['Age'].mean(), inplace = True)



# Fill missing Fare with mean

df_full['Fare'].fillna(df_full['Fare'].mean(), inplace = True)



# Fill missing Embarked with mode

df_full['Embarked'].fillna(df_full['Embarked'].mode()[0], inplace = True)
df_full['Fare_b'] = pd.qcut(df_full['Fare'], 5)

df_full['Age_b'] = pd.cut(df_full['Age'].astype(int), 5)
del [df_full['Name'], df_full['Cabin'], df_full['Ticket']]
df_full.head()
df_full.isnull().sum().plot(kind='bar')
for col in df_full[['Survived', 'Pclass', 'Fare_b', 'Embarked', 'Title', 'Age_b', 'Sex']]:

    sns.catplot(x=col, kind="count", data=df_full,

    height=5, 

    aspect=2)
sns.countplot(x='Pclass', hue="Survived", data=df_full)
sns.countplot(x='Fare_b', hue="Survived", data=df_full)
sns.countplot(x='Embarked', hue="Survived", data=df_full)
sns.countplot(x='Title', hue="Survived", data=df_full)
sns.countplot(x='Age_b', hue="Survived", data=df_full)
sns.countplot(x='Sex', hue="Survived", data=df_full)
# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(df_full.corr(), cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Generate dummies

df_full = pd.get_dummies(df_full, columns=['Embarked', 'Sex', 'Fare_b', 'Age_b', 'Title'],drop_first=True)

del [df_full['Age'], df_full['Fare']]



# Split train and test

df_train = df_full[df_full['type'] == 'train']

df_test = df_full[df_full['type'] == 'test']



del [df_train['PassengerId'], df_test['Survived'], df_train['type'], df_test['type']]



# separate the target variable

X_train = df_train.loc[:, df_train.columns != 'Survived']

y_train = df_train['Survived']
df_train.head()
sc = StandardScaler()

train_std = sc.fit_transform(X_train)

test_std = sc.fit_transform(df_test.loc[:, df_test.columns != 'PassengerId'])
train_std[0]
regressor = RandomForestClassifier(n_estimators=20, random_state=0)

regressor.fit(train_std, y_train)
final_pred = regressor.predict(test_std)
# Creates a dataframe with PassengerId and the predicted values



df_solution = pd.DataFrame()

df_solution['PassengerId'] = df_test['PassengerId']

df_solution['Survived'] = final_pred.astype(int)
df_solution['Survived'].value_counts()