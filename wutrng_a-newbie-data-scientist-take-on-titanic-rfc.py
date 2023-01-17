# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.info())
print('#'*30)
print(test.info())
# variable for test set 'PassengerId' needed for submission
passengerId = test['PassengerId']

# combine train and test set
titanic = train.append(test, ignore_index=True, sort=False )

# indexes for train and test set for modeling
train_idx = len(train)
test_idx = len(titanic) - len(test)
# stats summary
train.describe()
# for correlation heatmap reassign female: 1 and male: 0
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})

# heatmap correlation to 'Survived'
corr = train.corr()
idx = corr.abs().sort_values(by='Survived', ascending=False).index
corr_idx = train.loc[:,idx]
train_corr = corr_idx.corr()
mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(8,4))
sns.heatmap(train_corr, mask=mask, annot=True, cmap='seismic')
plt.figure(figsize=(4,4))
train['Survived'].value_counts().plot.pie(autopct= '%1.1f%%', cmap='Pastel1')
train['Sex'] = train['Sex'].map({1:'female', 0:'male'})
titanic.groupby(['Pclass', 'Sex'])['Survived'].mean()
# countplot for 'Survived'
sns.catplot(x='Survived', hue='Sex', data=titanic, col='Pclass', kind='count', palette='seismic', height=4)
# swarmplot for 'Age'
sns.catplot(x='Survived', y='Age', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')
# swarmplot for 'Fare'
sns.catplot(x='Survived', y='Fare', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')
# swarmplot for 'SibSp'
sns.catplot(x='Survived', y='SibSp', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')
# swarmplot for 'Parch'
sns.catplot(x='Survived', y='Parch', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')
# missing values
missing = titanic.isnull().sum().sort_values(ascending=False)
pct = (titanic.isnull().sum()/titanic.isnull().count()).sort_values(ascending=False)*100
total_missing = pd.concat([missing, pct], axis=1, keys=['total','percent'])
total_missing[total_missing['total']>0]
# 'Fare' NaN value
titanic[titanic['Fare'].isnull()]
# stats summary of 'Fare with 'Pclass and Embarked' groupby
titanic.groupby(['Pclass', 'Embarked'])['Fare'].describe()
# replace with median fare from 'Pclass' 3
titanic.iloc[1043,9] = 8.05
# 'Embarked' NaN value
titanic[titanic['Embarked'].isnull()]
# replace with 'C' as passengers' fare is closest to first class median price from 'Embarked' C
titanic.iloc[61,11] = 'C'
titanic.iloc[829, 11] = 'C'
# deeper look at 'Age' NaN values
#titanic[titanic['Age'].isnull()].sort_values(by='Name', ascending=True)
# deeper look at $0.00 fare
titanic[titanic['Fare'] == 0]
# median age for passengers with fare $0.00
work_median_age = titanic[(titanic['Fare'] == 0) & (titanic['Ticket'] != 'LINE')]['Age'].median()
work_median_age
# function to replace 'Age' NaN values for passengers with $0.00 fare
def workers_age(col):
    Age = col[0]
    Fare = col[1]
    if pd.isnull(Age):
        if Fare == 0:
            return work_median_age
    else:
        return Age
# apply function 
titanic['Age'] = titanic[['Age', 'Fare']].apply(workers_age, axis=1)
# to replace 'Age' NaN value for the remaining passengers, need to extract social class title from 'Name'
titanic['Title'] = titanic['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# new 'Title' column
titanic['Title'].value_counts()
# title dictionary to combine similar and rare titles together
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
# map title dictionary
titanic['Title'] = titanic['Title'].map(Title_Dictionary)
# groupby 'Pclass, Sex, and Title' to get age stats summary
titanic.groupby(['Pclass', 'Sex', 'Title'])['Age'].describe()
# deeper look at under 18 passengers
u_18 = titanic[titanic['Age']<=18]
#u_18
# 'Master' median age
master_median_age = u_18[u_18['Title'] == 'Master']['Age'].median()
# function to replace 'Master' median age
def master_age(col):
    Age = col[0]
    Title = col[1]
    if pd.isnull(Age):
        if Title == 'Master':
            return master_median_age
    else:
        return Age
# apply 'master_age' function
titanic['Age'] = titanic[['Age', 'Title']].apply(master_age, axis=1)
# deeper look at under 18 female passengers
#u_18[u_18['Sex'] == 'female']
# function to rename 'Miss' to 'Missy' to represent girls under 15
def girls_title(df):
    if df['Title'] == 'Miss':
        if df['Age'] < 15:
            return 'Missy'
        else:
            return df['Title']
    else:
        return df['Title']
# apply function 'girls_title'
titanic['Title'] = titanic[['Title', 'Age']].apply(girls_title, axis=1)

# groupby 'Pclass, Sex, and Title' updated stats summary for 'Age'
median_pclass_age = titanic.groupby(['Pclass', 'Sex', 'Title'])
#median_pclass_age['Age'].describe()
# lambda function to fill in remaining NaN values based on median age from 'median_pclass_age'
titanic['Age'] = median_pclass_age['Age'].apply(lambda x: x.fillna(x.median()))
# fill in 'Cabin' NaN values with 'U' for unknown
titanic['Cabin'] = titanic['Cabin'].fillna('U')
# verify missing values, only 'Survived' should have missing values
titanic.isnull().sum().sort_values(ascending=False)
# plots for continuous 'Age' and 'Fare'
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.distplot(titanic['Age'], ax=axes[0])
sns.distplot(titanic['Fare'], ax=axes[1])
# groupby ticket and count passengers traveling on same ticket
titanic['Same_Ticket'] = titanic.groupby('Ticket')['PassengerId'].transform('count')

# count the number of passengers traveling in a group
titanic[titanic['Same_Ticket'] >1]['Same_Ticket'].count()
# divide 'Fare' by 'Same_Ticket'
titanic['Fare'] = titanic['Fare'] / titanic['Same_Ticket']

# np.log1p 'Fare' to normalize
titanic['Fare_log1p'] = np.log1p(titanic['Fare'])

# updated distribution plots for fare
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.distplot(titanic['Fare'], ax=axes[0])
sns.distplot(titanic['Fare_log1p'], ax=axes[1])
# new 'Family' to represent passengers traveling with family or not
titanic['Family'] = titanic['SibSp'] + titanic['Parch']

# new 'Family_size' to represent total number of family members, if equal 1, passenger is traveling alone or with non family group
titanic['Family_size'] = titanic['Parch'] + titanic['SibSp'] + 1
#titanic['Family_size'].value_counts()
# deeper look at passengers with no family
no_family = titanic[(titanic['SibSp'] == 0) & (titanic['Parch'] ==0)]

# groupby to count number of passengers with same ticket and no family members
no_family['Friends_group'] = no_family.groupby('Ticket')['PassengerId'].transform('count')

# add 'Family' and 'Friends_group' to get group size
no_family['Group_size'] = no_family['Family'] + no_family['Friends_group']
# update titanic dataset with 'no_family'
nf = no_family[['PassengerId', 'Group_size']]

# create 'Group_size' from 'Family_size'
titanic['Group_size'] = titanic['Family_size']

# update titanic with 'no_family' data
new_df = titanic[['PassengerId', 'Group_size']].set_index('PassengerId')
new_df.update(no_family.set_index('PassengerId'))
titanic['Group_size'] = new_df.values
titanic['Group_size'] = titanic['Group_size'].astype(int)
# clean 'Ticket' by extracting letters and converting digit only tickets to 'xxx'
tickets = titanic['Ticket'].apply(lambda t: t.split('.')[0].split()[0].replace('/','').replace('.',''))

# convert to list
tickets = tickets.tolist()
# function to convert digit only tickets to 'xxx'
def ticket_digits(t):
    v = []
    for i in t:
        if i.isnumeric():
            i == 'xxx'
            v.append(i)
        else:
            v.append(i)
    return v
# call 'ticket_digits' function
tickets = ticket_digits(tickets)

# assign to titanic dataset
titanic['Ticks'] = pd.DataFrame(tickets)

# number of clean tickets 
ticket_count = dict(titanic['Ticks'].value_counts())
titanic['Ticket_count'] = titanic['Ticks'].apply(lambda t: ticket_count[t])
# extract surnames from 'Name'
titanic['Surname'] = titanic['Name'].apply(lambda x: x.split(',')[0].strip())

# create 'SurnameId' to group same surname
titanic['SurnameId'] = titanic.groupby('Surname').ngroup().add(1)

# groupby 'Ticket' and 'Surname' to represent groups with same ticket or family
titanic['GroupId'] = titanic.groupby(['Ticket', 'Surname']).ngroup().add(1)

# extract 'Cabin' letters to group
titanic['Cabin_group'] = titanic['Cabin'].apply(lambda x: x[0])
# separate dataframe to calculate confidence
group_survival = titanic[['Pclass', 'Survived', 'Surname', 'SurnameId', 'Group_size', 'GroupId', 'Family_size', 'Ticket']]

# sum the number of survivors in a group
group_survival['group_survived'] = group_survival.groupby('GroupId')['Survived'].transform('sum')

# adjust the number of survivors in a group
group_survival['adj_survived'] = group_survival['group_survived'] - group_survival['Survived'].apply(lambda x: 1 if x == 1 else 0)

# sum the number of dead in a group
group_survival['group_dead'] = group_survival.groupby('GroupId')['Survived'].transform('count') - group_survival.groupby('GroupId')['Survived'].transform('sum')

# adjust the number of dead in a group
group_survival['adj_dead'] = group_survival['group_dead'] - group_survival['Survived'].apply(lambda x: 1 if x == 0 else 0)

# confidence of survival on single group of passengers
no_data = (group_survival['Group_size'] - group_survival['adj_survived'] - group_survival['adj_dead'])/(group_survival['Group_size'])

# calculate confidence
confidence = 1 - no_data
group_survival['confidence'] = confidence * ((1/group_survival['Group_size']) * (group_survival['adj_survived'] - group_survival['adj_dead']))

# assign back to titanic
titanic['confidence'] = group_survival['confidence']
# plot for 'Ticks'
plt.figure(figsize=(10,4))
sns.barplot(x= 'Ticks', y='Survived', data=titanic[titanic['Ticket_count']>10])
plt.axhline(y = np.mean(titanic.groupby('Ticks')['Survived'].mean()), linestyle='-.')
# plots for 'Family_size', 'Group_size', and 'Cabin_group'
fig, axes = plt.subplots(1,3, figsize=(16,4))
sns.barplot(x='Family_size', y='Survived', data=titanic, ax=axes[0])
axes[0].axhline(y=np.mean(titanic.groupby('Family_size')['Survived'].mean()), linestyle='-.')
sns.barplot(x='Group_size', y='Survived', data=titanic, ax=axes[1])
axes[1].axhline(y=np.mean(titanic.groupby('Group_size')['Survived'].mean()), linestyle='-.')
sns.barplot(x='Cabin_group', y='Survived', data=titanic, ax=axes[2])
axes[2].axhline(y=np.mean(titanic.groupby('Cabin_group')['Survived'].mean()), linestyle='-.')
# add column for 'Kid'
titanic['Kid'] = (titanic['Age'] < 15).astype(int)
# function to categorize 'Family_size'
def family_2_cat(df):
    if df <= 2:
        return 'single'
    elif (df > 2) & (df < 5):
        return 'small'
    elif df >= 5:
        return 'large'     
# apply function on 'Family_size'
titanic['Family_cat'] = titanic['Family_size'].apply(family_2_cat)
# bin 'Age' to range
pd.cut(titanic['Age'], 5).value_counts()
# function to categorize 'Age'
def age_2_cat(df):
    if df < 15:
        return 'kid'
    elif (df >= 15) & (df <= 32):
        return 'young adult'
    elif (df > 32) & (df <= 64):
        return 'adult'
    elif (df > 64):
        return 'senior'
# apply function 'age_2_cat'
titanic['Age_range'] = titanic['Age'].apply(age_2_cat)
# bin 'Fare' to a range
titanic['Fare_range'] = pd.qcut(titanic['Fare'],3, labels=False)
# select best and worst survival chance from 'Ticks'
titanic['PC'] = (titanic['Ticks'] == 'PC').astype(int)
titanic['CA'] = (titanic['Ticks'] == 'CA').astype(int)

# select best and worst survival chance from 'Cabin_group'
titanic['D'] = (titanic['Cabin_group'] == 'D').astype(int)
titanic['U'] = (titanic['Cabin_group'] == 'U').astype(int)
# Feature correlation heatmap sorted by most correlated to "Survived"
corr = titanic.corr()
idx = corr.abs().sort_values(by='Survived', ascending=False).index
train_corr_idx = titanic.loc[:, idx]
train_corr = train_corr_idx.corr()
mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(20,10))
sns.heatmap(train_corr, mask=mask, annot =True, cmap = 'seismic')
# select 
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title','Group_size', 'CA', 'PC','Kid','confidence', 'Fare_log1p']

titanic_full = titanic[features]

# map female to 0, male to 1
titanic_full['Sex'] = titanic_full['Sex'].map({'female': 0, 'male': 1})

# get dummy variables
titanic_feats = pd.get_dummies(titanic_full)
# assign to train and test set
df_train = titanic_feats[:train_idx]
df_test = titanic_feats[test_idx:]

# assign for train test split
X = df_train
y = train['Survived']
test_X = df_test
# import necessary modeling libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# scale continuous data
scaler = MinMaxScaler()

# fit, tranform on X and transform on test_X
X[['Fare_log1p','Group_size']] = scaler.fit_transform(X[['Fare_log1p', 'Group_size']])
test_X[['Fare_log1p', 'Group_size']] = scaler.transform(test_X[['Fare_log1p','Group_size']])
# train split test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 8)
# models
rfc = RandomForestClassifier()
svc = SVC()
knn = KNeighborsClassifier()
gboost = GradientBoostingClassifier()
logreg = LogisticRegressionCV()

models = [rfc, svc, knn, gboost, logreg]
for model in models:
    print('cross validation of: {0}'.format(model.__class__))
    score = cross_val_score(model, x_train, y_train, cv= 5, scoring = 'accuracy')
    print('cv score: {0}'.format(np.mean(score)))
    print('#'*50)
# RFC
rfc = RandomForestClassifier(oob_score=True)

# fit
rfc.fit(x_train, y_train)

# oob_score_
print(rfc.oob_score_)

# model score
print(rfc.score(x_train,y_train))

# prediction on x_test
y_pred = rfc.predict(x_test)

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# accuracy score
print('model accuracy: ',accuracy_score(y_test,y_pred))

# train error by RMSE
print('train error rmse: ',np.sqrt(mean_squared_error(y_train, rfc.predict(x_train))))
# features of importance plot
feats = pd.DataFrame()
feats['feats'] = x_train.columns
feats['importance'] = rfc.feature_importances_
feats.sort_values(by='importance', ascending=True, inplace=True)
feats.set_index('feats', inplace=True)
feats.plot(kind='barh')
rfc_submit = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc.predict(test_X)})
rfc_submit.to_csv('rfc_submit.csv', index=False)
# Optimize RFC parameters with GridSearchCV
model = RandomForestClassifier()

# parameters 
parameters = {
    "n_estimators": [50,100,200,300,400,500],
    "max_depth": [i for i in range(2,8)], 
    "min_samples_leaf": [i for i in range(2,8)],
    "max_leaf_nodes": [i for i in range(6,12)],
    "bootstrap": [True],
    'oob_score': [True],
    'max_features': [1,2,3]
}

# GridSearchCV (kaggle notebook reason will comment out)
#grid = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', n_jobs=-1, verbose=1, cv=5)
# fit x_train, y_train, for 
# grid.fit(x_train,y_train)
# print('best estimator: ', grid.best_estimator_)
# print('best params: ', grid.best_params_)
best_estimator = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features=3, max_leaf_nodes=7,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=50,
            n_jobs=None, oob_score=True, random_state=None, verbose=0,
            warm_start=False)

best_params = {'bootstrap': True, 'max_depth': 6, 'max_features': 3, 'max_leaf_nodes': 7, 'min_samples_leaf': 2, 'n_estimators': 50, 'oob_score': True}
    
rfc_grid = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features=3, max_leaf_nodes=7,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=50,
            n_jobs=None, oob_score=True, random_state=None, verbose=0,
            warm_start=False)
warnings.filterwarnings('ignore')
rfc_grid.fit(x_train, y_train)
print('oob score: ', rfc_grid.oob_score_)
print('accuracy score on x_test: ',accuracy_score(y_test, rfc_grid.predict(x_test)))
# optimized RFC prediction
grid_prediction = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc_grid.predict(test_X)})
grid_prediction.to_csv('prediction.csv', index=False)
