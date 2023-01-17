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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, ShuffleSplit

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFECV

from sklearn import model_selection



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



raw_train = pd.read_csv("../input/train.csv")

raw_test = pd.read_csv("../input/test.csv")

raw_train.head()
raw_test.describe()
print("----- train -----")

print(raw_train.isnull().sum())



print("----- test -----")

print(raw_test.isnull().sum())
raw_train.head()
from sklearn.ensemble import RandomForestClassifier



def evaluate_rf(df):

    

    cv = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6)

    

    Y_train = df["Survived"]

    X_train = df.drop(labels = ["Survived"],axis = 1)

    

    classifier = RandomForestClassifier()

    cv_result = cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = cv, n_jobs=4)

    

    print("accuracy mean: ", cv_result.mean())

    print("         std:  ", cv_result.std())



evaluate_rf(raw_train[['Survived', 'Pclass', 'SibSp', 'Parch']])
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# from sklearn.ensemble import AdaBoostClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb



# prepare data

def evaluate(df):

    

    cv = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

    

    Y_train = df["Survived"]

    X_train = df.drop(labels = ["Survived"],axis = 1)

    

    algorithms = [

        "SV", "RandomForest", "GradientBoosting", "ExtraTrees",

        "LinearDiscriminant", "QuadraticDIscriminant",

        "DecisionTree", "KNeighbors", "LogisticRegression", "MLPClassifier", "GaussianNB",

        "XGBClassifier"

    ]

    

    classifiers = [

        SVC(),



        RandomForestClassifier(),

        GradientBoostingClassifier(),

        ExtraTreesClassifier(),



        LinearDiscriminantAnalysis(),

        QuadraticDiscriminantAnalysis(),



        DecisionTreeClassifier(),

        KNeighborsClassifier(),

        LogisticRegression(),

        MLPClassifier(),

        GaussianNB(),

        

        xgb.XGBClassifier(),

    ]

    

    cv_results = []

    for classifier in classifiers :

        cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = cv, n_jobs=4))



    cv_means = []

    cv_std = []

    for cv_result in cv_results:

        cv_means.append(cv_result.mean())

        cv_std.append(cv_result.std())



    cv_res = pd.DataFrame({

        "CrossValMeans":cv_means,

        "CrossValerrors": cv_std,

        "Algorithm":algorithms})



    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

    g.set_xlabel("Mean Accuracy")

    g = g.set_title("Cross validation scores")



    r = list(zip(algorithms, cv_means))

    r.sort(key=lambda x: x[1])

    max_result = r[-1]

    print("Max Accuracy: ", max_result)



evaluate(raw_train[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']])
sns.heatmap(raw_train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True)

# raw_train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.countplot('Pclass', hue='Survived', data=raw_train)

sns.factorplot('Pclass', 'Survived', data=raw_train)

raw_train[['Pclass', 'Survived', 'Sex']].groupby(['Pclass', 'Sex']).mean()
raw_train[raw_train.Cabin.isnull()].Pclass.value_counts()
sns.countplot('Sex',hue='Survived', data=raw_train)

sns.factorplot('Sex', 'Survived', data=raw_train)

raw_train[['Sex', 'Survived']].groupby('Sex').mean()
raw_train.Name.head()
def name2title(name):

    title = name.split(".")[0].split(" ")[-1]



    if title in ['Countess', 'Lady', 'Mlle', 'Mme']:

        title = 'Mrs'



    if title in ['Ms']:

        title = 'Miss'

    

    if title not in ['Mr', 'Miss', 'Mrs', 'Master']:

        title = 'Other'



    return title



def add_title(df_):

    df = df_.copy()

    df['Title'] = df.Name.map(name2title)

    return df



_df = add_title(raw_train)

print(_df.Title.value_counts())

_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex']).agg(['size', 'mean', 'std'])
sns.distplot(raw_train.Age.dropna())
sns.distplot(raw_train[raw_train.Survived == 1].Age.dropna(), label='Survived')

sns.distplot(raw_train[raw_train.Survived == 0].Age.dropna(), label='Dead')

plt.legend()
raw_train[raw_train.Age.isnull()]
def assign_age_by_median(df_):

    df = df_.copy()

    df.Age = df.Age.fillna(df.Age.median())

    return df



def assign_age_by_title(df_):

    df = df_.copy()

    title_age = df.groupby('Title').Age.mean()

    df = df.assign(

        Age = df.apply(lambda x: x.Age if pd.notnull(x.Age) else title_age[x.Title] , axis=1)

    )

    df.Age = df.Age.fillna(df.Age.median())  # when is null

    return df



# _df = assign_age_by_median(_df)

_df = add_title(raw_train)

_df = assign_age_by_title(_df)

sns.distplot(_df.Age)

_df.Age.isnull().sum()
sns.distplot(raw_train.Fare)
_ , ax = plt.subplots(figsize =(20, 10))

sns.distplot(raw_train[raw_train.Survived == 1].Fare.dropna(), label='Survived')

sns.distplot(raw_train[raw_train.Survived == 0].Fare.dropna(), label='Dead')

plt.legend()
raw_train[['SibSp', 'Survived']].groupby(['SibSp']).agg(['size', 'mean', 'std'])
raw_train[['Parch', 'Survived']].groupby(['Parch']).agg(['size', 'mean', 'std'])
_df = raw_train.copy()

_df['FamilySize'] = _df.SibSp + _df.Parch

_df[['FamilySize', 'Survived']].groupby(['FamilySize']).agg(['size', 'mean', 'std'])
def preprocess(df_):

    df = df_.copy()

    le = LabelEncoder()



    df = add_title(df)

    df = assign_age_by_title(df)

    

    df.Title = le.fit_transform(df.Title)

    df.Sex = le.fit_transform(df.Sex)

    

    df.Age = le.fit_transform(pd.cut(df.Age, 6))

    df.Fare = df.Fare.fillna(df.Fare.median())

    df.Fare = le.fit_transform(pd.qcut(df.Fare, 4))



    df.Embarked = df.Embarked.fillna("S")

    df.Embarked = le.fit_transform(df.Embarked)



    df['FamilySize'] = df.SibSp + df.Parch

    df['IsAlone'] = df.FamilySize == 0

    

    # df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title', 'Embarked', 'FamilySize'], axis=1)



    return df



data_train = preprocess(raw_train)

data_train.head()
data_train = preprocess(raw_train)

evaluate(data_train)
def predict(train, test):



    y_train = train["Survived"]

    x_train = train.drop(labels = ["Survived"],axis = 1)

    



    xgbc = xgb.XGBClassifier()

    xgbc.fit(x_train, y_train)

    

    y_test = xgbc.predict(test)

    

    return y_test



data_train = preprocess(raw_train)

data_test = preprocess(raw_test)

y_test = predict(data_train, data_test)

result = raw_test.copy()

result['Survived'] = y_test

result.to_csv('xgb_result.csv')

result[['PassengerId', 'Survived']].to_csv('submit01.csv', index=False)

result.head()
df = preprocess(raw_train)

_ , ax = plt.subplots(figsize =(20, 16))

sns.heatmap(df.corr(), annot=True)
data_train = preprocess(raw_train)

evaluate(data_train[['Survived', 'Sex']])