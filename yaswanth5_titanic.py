# All Basic Imports

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold



import re as re

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
# reading train and test data

train_df =  pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
test_df.head()
train_df.info()

# there are few missing values in Age, Cabin, Embarked.

# so need to find corelation with other columns to fill these values accordinigly
test_df.info()

# age, cabin, fare missing
train_df.describe()
test_df.describe()
#survived count

sns.countplot('Survived', data=train_df)

print(train_df['Survived'].value_counts())

print(train_df['Survived'].value_counts(normalize=True))
#survived count w.r.f to sex

sns.countplot('Survived', hue='Sex', data=train_df)

# femals has most chance of survival



#male survived count and percentage

print(train_df[train_df['Sex'] == 'male']['Survived'].value_counts())

print(train_df[train_df['Sex'] == 'male']['Survived'].value_counts(normalize=True))



#female survived count and percentage

print(train_df[train_df['Sex'] == 'female']['Survived'].value_counts())

print(train_df[train_df['Sex'] == 'female']['Survived'].value_counts(normalize=True))
#relation between Age and Pclass

plt.figure(figsize=(10,7))

sns.boxplot('Pclass', 'Age', data=train_df)

#can conclude that to reach first class age range is 30-50,

#so when to fill null values of age, pclass need to be take into consideration
sns.countplot('Survived', hue='Embarked', data=train_df)
sns.countplot('Embarked', hue='Pclass', data=train_df)

# for pclass 1 more chances of embarked is 'S' and vice versa
# mean value of ages depends on pclass

print(train_df[train_df['Pclass'] == 1]['Age'].mean())

print(train_df[train_df['Pclass'] == 2]['Age'].mean())

print(train_df[train_df['Pclass'] == 3]['Age'].mean())
print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
def includeFamilySize(df):

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['isAlone'] = 0

    df.loc[df['FamilySize'] == 1, 'isAlone'] = 1
includeFamilySize(train_df)

includeFamilySize(test_df)
print (train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

print (train_df[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean())
def immute_ages(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        if Pclass == 2:

            return 30

        else:

            return 25

    else:

        return Age
# to check values of pclass where embarked is null

train_df[pd.isnull(train_df['Embarked'])]
# fill fare data in test data acc to corresponding pclass

# fare depends on pclass

plt.figure(figsize=(20,7))

sns.boxplot('Pclass','Fare', data=test_df)
test_df[test_df['Fare'].isnull()]

print(test_df[test_df['Pclass'] == 3]['Fare'].mean())

test_df['Fare'].fillna(12.45, inplace=True)
def fare_age_Categ(df):

    df['CategoricalFare'] = pd.qcut(df['Fare'], 4)

    df['CategoricalAge'] = pd.cut(df['Age'], 5)
fare_age_Categ(train_df)

fare_age_Categ(test_df)

print (train_df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

print (train_df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""
def addTitle(df):

    df['Title'] = df['Name'].apply(get_title)
addTitle(train_df)

addTitle(test_df)
print(pd.crosstab(train_df['Title'], train_df['Sex']))
def compressTitleData(df):

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')
compressTitleData(test_df)

compressTitleData(train_df)
print (train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
def getNameLength(df):

    df['NameLength'] = df['Name'].apply(len)
getNameLength(train_df)

getNameLength(test_df)
def getCabinStatus(df):

    df['Has_Cabin'] = df['Cabin'].apply(lambda x:0 if type(x) == float else 1 )
getCabinStatus(train_df)

getCabinStatus(test_df)
def cleanDataFrame(df):

    # fill null ages to mean value of age, depends on pclass

    df['Age'] = df[['Age', 'Pclass']].apply(immute_ages, axis=1)

    

    # fill embarked acc to pclass

    df['Embarked'].fillna('S', inplace=True)

    

    # Mapping Sex

    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    df['Title'] = df['Title'].map(title_mapping)

    df['Title'] = df['Title'].fillna(0)

    

    # Mapping Embarked

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)

    

    # Mapping Age

    df.loc[ df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age'] = 4

    df['Age'] = df['Age'].astype(int)
cleanDataFrame(train_df)

cleanDataFrame(test_df)
def fetureSelection(df):

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'CategoricalAge', 'CategoricalFare']

    df.drop(drop_elements, axis=1, inplace=True)
fetureSelection(train_df)

fetureSelection(test_df)
print(test_df.columns)

print(train_df.columns)
train_df.head()
plt.figure(figsize=(14,12))

sns.heatmap(train_df.astype(float).corr(), annot=True)
train_x = train_df.drop(['Survived'], axis=1)

train_y = train_df['Survived']

test_x = test_df
train_x.head()
test_x.head()
def predict(train_x, train_y, test_df, algo):

    algo.fit(train_x, train_y)

    pred_y = algo.predict(test_df)

    score = algo.score(train_x, train_y)

    return pred_y, score
# logistic regression

lr_pred_train, score = predict(train_x, train_y, train_x, LogisticRegression())

lr_pred_test, score = predict(train_x, train_y, test_x, LogisticRegression())

score
# decision tree



dt_pred_train, score = predict(train_x, train_y, train_x, DecisionTreeClassifier(max_depth=10, min_samples_split=5,random_state=1))

dt_pred_test, score = predict(train_x, train_y, test_x, DecisionTreeClassifier(max_depth=10, min_samples_split=5,random_state=1))

score
#Random forest

rfc = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=600, random_state=1)

rt_pred_train, score = predict(train_x, train_y, train_x, rfc)

rt_pred_test, score = predict(train_x, train_y, test_x, rfc)

score
#svm

svc_model = SVC()

svc_pred_train, score = predict(train_x, train_y, train_x, svc_model)

svc_pred_test, score = predict(train_x, train_y, test_x, svc_model)

score
#grid search

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(),param_grid,refit=True)

grid_pred_train, score = predict(train_x, train_y, train_x, grid)

grid_pred_test, score = predict(train_x, train_y, test_x, grid)

score
#Kneibhours

knn_pred_train, score = predict(train_x, train_y, train_x, KNeighborsClassifier())

knn_pred_test, score = predict(train_x, train_y, test_x, KNeighborsClassifier())

score
retest = pd.read_csv('test.csv')
prediction_y = np.around(((rt_pred_test + 2*knn_pred_test + svc_pred_test + 0.01)/4)).astype(int)
final_df = pd.DataFrame({'PassengerId': retest['PassengerId'], 'Survived': svc_pred_test})

final_df.to_csv('rf.csv', index=False)