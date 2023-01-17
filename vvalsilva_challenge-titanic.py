import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv(dirname+"/train.csv")

df_test = pd.read_csv(dirname+"/test.csv")

df_genderSubmission = pd.read_csv(dirname+"/gender_submission.csv")
df_train.head()
df_test.head()
df_genderSubmission.head()
df_train.info()
df_test.info()
df_train['Survived'].value_counts(normalize = True)
sns.countplot(df_train['Survived'])
df_train.isna().sum()
df_test.isna().sum()
ids = df_test.PassengerId

df_train = df_train.drop(['Ticket','Cabin','Name'], axis=1)

df_test = df_test.drop(['Ticket','Cabin','Name'], axis=1)
df_train.head()
age_median = df_train['Age'].median()

df_train['Age'].fillna(value = age_median, inplace=True)
age_median = df_test['Age'].median()

df_test['Age'].fillna(value = age_median, inplace=True)
median_fare = df_test['Fare'].median()

df_test['Fare'].fillna(value=median_fare, inplace=True)
df_train.head()
df_train.describe()
df_test.describe()
fig, axes = plt.subplots(2,1, figsize = (10,10))

sns.heatmap(df_train.corr(), cmap = 'Spectral', annot = True, ax=axes[0])

temp = df_train.corrwith(df_train['Survived']).sort_values().drop('Survived')

sns.barplot(temp.index,temp.values,ax=axes[1]);
sns.distplot(df_train["Age"], kde=True, hist=True)

plt.title('Age Distribution', fontsize= 15)

plt.ylabel("Density", fontsize= 15)

plt.xlabel("Age", fontsize= 15)

plt.show();

survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



female = df_train[df_train['Sex']=='female']

male = df_train[df_train['Sex']=='male']



ax = sns.distplot(female[female['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(female[female['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')





ax = sns.distplot(male[male['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(male[male['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

ax.set_title('Male')
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
g = sns.catplot(x="Sex", y="Survived", col="Pclass",

                data=df_train, saturation=.5,

                kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survived")

  .set_xticklabels(["Male", "Female"])

  .set_titles("{col_name} {col_var}")

  .set(ylim=(0, 1))

  .despine(left=True))
var_objects =  df_train.select_dtypes('object').columns

var_objects
df_train = pd.get_dummies(df_train,columns=["Sex","Embarked"])

df_test = pd.get_dummies(df_test,columns=["Sex","Embarked"])

df_test.head()
df_train.head()
Y = df_train.Survived

X = df_train.drop(columns=["Survived"])

X.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

df_test = sc.transform(df_test)

print(X.shape,df_test.shape)
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,random_state=2)

print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

kfold_5 = KFold(shuffle = True, n_splits = 10)
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

modelRF.fit(X_train,Y_train)

print("%.4f" % modelRF.oob_score_)
from sklearn.metrics import classification_report



Y_pred = modelRF.predict(X_val)

print(classification_report(Y_val,Y_pred))
Y_pred = modelRF.predict(df_test)

output = pd.DataFrame({'PassengerId': ids, 'Survived': Y_pred})

output.to_csv('survived.csv', index=False)