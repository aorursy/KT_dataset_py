# Data Analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

from scipy import stats

from sklearn.preprocessing import LabelEncoder



# Other Imports

import re

import warnings



# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

%matplotlib inline

# Machine Learning

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import metrics

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
# Loading the datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.info()
# Null values train dataset

train_df.isna().sum()
# Null values test dataset

test_df.isna().sum()
train_df['source'] = 'train'

test_df['source'] = 'test'
dataset = pd.concat([train_df, test_df], ignore_index=True)
dataset.isnull().sum()
train_df.isnull().sum()
plt.figure(figsize=(10,4))

plt.subplot(121)

sns.distplot(train_df['Age'].dropna())

plt.title('Age Distribution')



plt.subplot(122)

sns.distplot(train_df['Fare'].dropna())

plt.title('Fare Distribution')
train_df['Survived'].value_counts()
train_df.Parch.value_counts()
train_df.SibSp.value_counts()/891*100
train_df.Pclass.value_counts()/train_df.shape[0]*100
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(sns.distplot, 'Age', bins=20, kde=False, color='green')
grid_pivot1 = train_df.pivot_table(columns='Survived', values='Age', aggfunc='mean')
grid_pivot1
sns.pointplot(x=train_df['Survived'].dropna(), y=train_df['Age'].dropna())
grid = sns.FacetGrid(train_df, col='Survived', row='Sex')

grid.map(sns.distplot, 'Age', bins=20, kde=False, color='green')
# sns.pointplot(x=train_df['Survived'].fillna(-1), y=train_df['Age'].fillna(-1), hue=train_df['Sex'].fillna(-1)) 
grid_pivot2 = train_df.pivot_table(index='Sex', columns='Survived', values='Age', aggfunc='mean')

grid_pivot2
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass')

grid.map(sns.distplot, 'Age', bins=20, kde=False,rug=True, color='red')
grid_pivot3 = train_df.pivot_table(index='Pclass',columns='Survived', values='Age')
grid_pivot3
grid = sns.FacetGrid(train_df, col='Survived', row='SibSp')

grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='pink')
grid = sns.FacetGrid(train_df, col='Survived', row='Parch')

grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='green')
grid = sns.FacetGrid(train_df, col='Survived', row='Embarked')

grid.map(sns.distplot,'Age', bins=20, kde=False,rug=True, color='black')
grid = sns.FacetGrid(train_df, row='Embarked')

grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')

grid.add_legend()
train_df.pivot_table(index=['Embarked','Pclass'], columns='Sex', values='Survived')
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived')

grid.map(sns.barplot, 'Sex', 'Fare', ci=None)
def extract_titles(name):

    tit = re.findall(' ([A-Za-z]+)\.', name)

    return tit[0]
dataset['Title'] = dataset['Name'].apply(lambda x: extract_titles(x))
# train_df['Title'] = train_df['Name'].apply(lambda x: extract_titles(x))



# test_df['Title'] = test_df['Name'].apply(lambda x: extract_titles(x))
dataset[dataset['source'] == 'train'].isnull().sum()
dataset[dataset['source'] == 'test'].isnull().sum()
dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr','Major', 'Lady', 'Sir','Col', 'Capt', 'Countess',

       'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace(['Ms', 'Mlle'], 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
dataset[dataset['source'] == 'train'].Title.value_counts()
dataset[dataset['source'] == 'train'].pivot_table(index='Title', values='Survived')
dataset[dataset['source'] == 'train'].pivot_table(index='Title', columns='Survived', values='Age')
# dataset[dataset['source'] == 'train'] = dataset[dataset['source'] == 'train'].drop(['Name','PassengerId'], axis=1)



# dataset[dataset['source'] == 'train'].isnull().sum()



# train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)



# test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
dataset[dataset['source'] == 'train'].head()
sex_dummy = pd.get_dummies(dataset['Sex'])
sex_dummy.shape
dataset.shape
# train_df = pd.concat([train_df, sex_dummy], sort=False)

dataset = dataset.join(sex_dummy)
dataset.head()
grid = sns.FacetGrid(dataset[dataset['source'] == 'train'], row='Pclass', col='Sex')

grid.map(sns.distplot, 'Age', bins=20, kde=False)
# train_df['Sex'] = train_df['Sex'].astype(int)
grid_pivot = dataset[dataset['source'] == 'train'].pivot_table(index='Pclass',columns='Sex', values='Age', aggfunc='median')
grid_pivot
# def fage(x):

#     age_med = grid_pivot.loc[x['Pclass'], x['Sex']]

#     return age_med
# dataset[dataset['source'] == 'train']['Age'].isna().sum()
# dataset.isna().sum()
# dataset['Age'].fillna(dataset[dataset['Age'].isnull()].apply(fage, axis=1), inplace=True)
guess_ages = np.zeros((2,3))

guess_ages



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = dataset[(dataset['Sex'] == i) & \

                              (dataset['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()

        

        # Convert random age float to nearest .5 age

        #guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



for i in range(0, 2):

    for j in range(0, 3):

        dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                'Age'] = guess_ages[i,j]
dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
dataset.AgeBand.value_counts()
dataset['AgeBand'] = dataset['AgeBand'].astype(str)
dataset.loc[dataset['source'] == 'train'].pivot_table(index='AgeBand', values='Survived')
# train_df.head()
# train_df.loc[train_df['Age'] <= 16, 'Age'] = 0

# train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1

# train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2

# train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3

# train_df.loc[(train_df['Age'] > 64), 'Age'] = 4

# train_df.head()
def ageclass(x):

    if x <= 16:

        return 0

    elif x > 16 and x <= 32:

        return 1

    elif x > 32 and x <= 48:

        return 2

    elif x > 48 and x <= 64:

        return 3

    else:

        x > 64

        return 4
dataset['AgeClass'] = dataset['Age'].apply(ageclass)
dataset.head()
dataset['Family_Size']  = dataset['Parch'] + dataset['SibSp'] + 1
dataset.pivot_table(index='Family_Size', values='Survived').sort_values(by='Survived',ascending=False )
def isalone(x):

    if x == 1:

        return 1

    else:

        return 0
dataset['IsAlone'] = dataset['Family_Size'].apply(isalone)
dataset[dataset['source']=='train'].pivot_table(index='IsAlone', values='Survived')
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
dataset.isnull().sum()
embarked_dummy = pd.get_dummies(dataset['Embarked'])
dataset = dataset.join(embarked_dummy)
dataset.head()
dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
le = LabelEncoder()
dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
dataset['FareClass'] = le.fit_transform(dataset['FareBand'])
dataset.pivot_table(index='FareClass', values='Survived').sort_values(by='Survived', ascending=False)
# def fareclass(x):

#     if x <= 7.896 :

#         return 0

#     elif x > 7.896 and x <= 14.454:

#         return 1

#     elif x > 14.454 and x <= 31.275 :

#         return 2

#     elif x > 31.275 and x <= 512.329:

#         return 3
# dataset['FareClass'] = dataset['Fare'].apply(fareclass)
# dataset['FareClass'] = dataset['Fare'].apply(fareclass)



# dataset['FareClass'] = dataset['FareClass'].astype(int)



# test_df['FareClass'] = test_df['Fare'].apply(fareclass).astype(int)
dataset.keys()
dataset['Title'] = le.fit_transform(dataset['Title'])
dataset['Age*Class'] = dataset['AgeClass'] * dataset['Pclass']
dataset.head()
drop_these = 'Age Cabin Embarked Fare Name Parch Ticket Sex SibSp AgeBand Family_Size FareBand'.split(' ')
drop_these
dataset = dataset.drop(drop_these, axis=1)
dataset.head()
train_cleaned = dataset[dataset['source'] == 'train']

test_cleaned = dataset[dataset['source'] == 'test']
train_cleaned['Survived'] = train_cleaned['Survived'].astype(int)
train_cleaned.keys()
'Pclass', 'Survived', 'source', 'Title', 'female','male', 'AgeClass', 'IsAlone', 'C', 'Q', 'S', 'FareClass', 'Age*Class'
def predict_model(dtrain, dtest, predictor, outcome, model):

    model.fit(dtrain[predictor], dtrain[outcome])

    dtrain_pred = model.predict(dtest[predictor])

    score = model.score(dtrain[predictor], dtrain[outcome])*100

    return score, dtrain_pred
predictors_var = ['Pclass','Title', 'female','male', 'AgeClass', 'IsAlone', 'C', 'Q', 'S', 'FareClass', 'Age*Class']

outcome_var = 'Survived'

# model_name = logreg

traindf = train_cleaned

testdf = test_cleaned
logreg = LogisticRegression()
predict_model(traindf, testdf, predictors_var, outcome_var, logreg)
coef1 = pd.Series(logreg.coef_[0], predictors_var).sort_values()
coef1.sort_values(ascending=False)
svc = SVC()
predict_model(traindf, testdf, predictors_var, outcome_var, svc)
coef2 = pd.Series(svc.coef_[0], predictors_var).sort_values()



coef2.sort_values(ascending=False)
knn = KNeighborsClassifier(n_neighbors=3)
predict_model(traindf, testdf, predictors_var, outcome_var, knn)
gaussian = GaussianNB()
predict_model(traindf, testdf, predictors_var, outcome_var, gaussian)
decision_tree = DecisionTreeClassifier()
predict_model(traindf, testdf, predictors_var, outcome_var, decision_tree)
coef4 = pd.Series(decision_tree.feature_importances_, predictors_var).sort_values()



coef4.sort_values(ascending=False)
random_forest = RandomForestClassifier(n_estimators=100)
predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)
predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)[1]
results = predict_model(traindf, testdf, predictors_var, outcome_var, random_forest)[1]
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": results

    })
submission.to_csv('submission_updated.csv', index=False)