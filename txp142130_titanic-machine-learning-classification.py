import re

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV



%matplotlib inline

sns.set_style('whitegrid')

pd.set_option('display.max_columns', None)
train_data = pd.read_csv('../input/train.csv', dtype={'Age': np.float64}, )

test_data = pd.read_csv('../input/test.csv', dtype={'Age': np.float64}, )
train_data.shape
train_data.describe(include='all')
test_data.shape
test_data.describe(include='all')
train_data.duplicated().sum()
percent_missing = train_data.isnull().sum()/ len(train_data) * 100

percent_missing
plt.figure(figsize=(10,10))

sns.heatmap(train_data.isnull(), cbar = False, cmap = 'YlGnBu')
train_data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
train_data['Title'] = train_data['Name'].apply(lambda x: (re.search(' ([a-zA-Z]+)\.', x)).group(1))

test_data['Title'] = test_data['Name'].apply(lambda x: (re.search(' ([a-zA-Z]+)\.', x)).group(1))



train_data['Title'].value_counts()
titles = {'Capt':       'Officer',

          'Col':        'Officer',

          'Major':      'Officer',

          'Jonkheer':   'Royalty',

          'Don':        'Royalty',

          'Sir' :       'Royalty',

          'Dr':         'Officer',

          'Rev':        'Officer',

          'Countess':   'Royalty',

          'Dona':       'Royalty',

          'Mme':        'Mrs',

          'Mlle':       'Miss',

          'Ms':         'Mrs',

          'Mr' :        'Mr',

          'Mrs' :       'Mrs',

          'Miss' :      'Miss',

          'Master' :    'Master',

          'Lady' :      'Royalty'

                    } 



for k,v in titles.items():

    train_data.loc[train_data['Title'] == k, 'Title'] = v

    test_data.loc[test_data['Title'] == k, 'Title'] = v



#New frequencies.

train_data['Title'].value_counts()
print(train_data.groupby(['Sex', 'Pclass', 'Title', ])['Age'].median())
train_data['Age'] = train_data.groupby(['Sex','Pclass','Title'])['Age'].apply(lambda x: x.fillna(x.median()))

test_data['Age'] = test_data.groupby(['Sex','Pclass','Title'])['Age'].apply(lambda x: x.fillna(x.median()))
p = sns.countplot(data=train_data, x='Sex')
train_data.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize=True)
train_data['Family'] = train_data['Parch'] + train_data['SibSp']

test_data['Family'] = test_data['Parch'] + test_data['SibSp']
train_data.groupby(['Family'])['Survived'].value_counts(normalize=True)
def FamilySize(x):

    """

    A function for Family size transformation

    """

    if x == 1 or x == 2:

        return 'little'

    elif x == 3:

        return 'medium'

    elif x >= 5:

        return 'big'

    else:

        return 'single'



train_data['Family'] = train_data['Family'].apply(lambda x : FamilySize(x))

test_data['Family'] = test_data['Family'].apply(lambda x : FamilySize(x))
train_data.groupby(['Pclass', 'Family'])['Survived'].mean()
def Ticket_Prefix(x):

    """

    Function for extracting prefixes. Tickets have length of 1-3.

    """

    l = x.split()

    if len(x.split()) == 3:

        return x.split()[0] + x.split()[1]

    elif len(x.split()) == 2:

        return x.split()[0]

    else:

        return 'None'



train_data['TicketPrefix'] = train_data['Ticket'].apply(lambda x: Ticket_Prefix(x))

test_data['TicketPrefix'] = test_data['Ticket'].apply(lambda x: Ticket_Prefix(x))
#There are many similar prefixes, but combining them doesn't yield a significantly better result.

train_data.TicketPrefix.unique()
ax = plt.subplot()

ax.set_ylabel('Average Fare')

train_data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(7, 4), ax=ax)

test_data['Fare'] = test_data.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))
train_data.Cabin.fillna('Unknown',inplace=True)

test_data.Cabin.fillna('Unknown',inplace=True)



train_data['Cabin'] = train_data['Cabin'].map(lambda x: x[0])

test_data['Cabin'] = test_data['Cabin'].map(lambda x: x[0])
#Now let's see. Most of the cabins aren't filled.

f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y='Cabin', data=train_data, color='B')
#Other cabins vary in number.

sns.countplot(y='Cabin', data=train_data[train_data.Cabin != 'U'], color='R')
#Factorplot shows that most people, for whom there is no info on Cabin, didn't survive.

sns.catplot('Survived', col='Cabin', col_wrap=4, data=train_data[train_data.Cabin == 'U'], kind='count', height=2.5, aspect=.8)
#For passengers with known Cabins survival rate varies.

sns.catplot('Survived', col='Cabin', col_wrap=4, data=train_data[train_data.Cabin != 'U'], kind='count', height=2.5, aspect=.8)
train_data.groupby(['Cabin']).mean()[train_data.groupby(['Cabin']).mean().columns[1:2]]
MedEmbarked = train_data.groupby('Embarked').count()['PassengerId']

train_data.Embarked.fillna(MedEmbarked, inplace=True)
#This is how the data looks like now.

train_data.head()
#Drop unnecessary columns

to_drop = ['Ticket', 'Name', 'SibSp', 'Parch']

for i in to_drop:

    train_data.drop([i], axis=1, inplace=True)

    test_data.drop([i], axis=1, inplace=True)
#Pclass in fact is a categorical variable, though it's type isn't object.

for col in train_data.columns:

    if train_data[col].dtype == 'object' or col == 'Pclass':

        dummies = pd.get_dummies(train_data[col], drop_first=False)

        dummies = dummies.add_prefix('{}_'.format(col))

        train_data.drop(col, axis=1, inplace=True)

        train_data = train_data.join(dummies)

for col in test_data.columns:

    if test_data[col].dtype == 'object' or col == 'Pclass':

        dummies = pd.get_dummies(test_data[col], drop_first=False)

        dummies = dummies.add_prefix('{}_'.format(col))

        test_data.drop(col, axis=1, inplace=True)

        test_data = test_data.join(dummies)
#This is how the data looks like now.

train_data.head()
X_train = train_data.drop('Survived',axis=1)

Y_train = train_data['Survived']

X_test  = test_data
clf = RandomForestClassifier(n_estimators = 15,

                                criterion = 'gini',

                                max_features = 'sqrt',

                                max_depth = None,                                

                                min_samples_split =7,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 18)

clf = clf.fit(X_train, Y_train)

indices = np.argsort(clf.feature_importances_)[::-1]



print('Feature ranking:')

for f in range(X_train.shape[1]):

    print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]], clf.feature_importances_[indices[f]]))
model = SelectFromModel(clf, prefit=True)

train_new = model.transform(X_train)

train_new.shape
best_features = X_train.columns[indices[0:train_new.shape[1]]]

X = X_train[best_features]

Xt = X_test[best_features]

best_features
X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.33, random_state=44)
plt.figure(figsize=(30,20))

#N Estimators

plt.subplot(3,3,1)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(n_estimators=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.axis('tight')

plt.title('N Estimators')
plt.figure(figsize=(30,20))

#Criterion

plt.subplot(3,3,2)

feature_param = ['gini','entropy']

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(criterion=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.title('Criterion')

plt.xticks(range(len(feature_param)), feature_param)

plt;
plt.figure(figsize=(30,20))

#Max Features

plt.subplot(3,3,3)

feature_param = ['auto','sqrt','log2',None]

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_features=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.axis('tight')

plt.title('Max Features')

plt.xticks(range(len(feature_param)), feature_param)

plt;
plt.figure(figsize=(30,20))

#Max Depth

plt.subplot(3,3,4)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_depth=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Max Depth')

plt;
plt.figure(figsize=(30,20))

#Min Weight Fraction Leaf

plt.subplot(3,3,6)

feature_param = np.linspace(0,0.5,10)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Min Weight Fraction Leaf')
plt.figure(figsize=(30,20))

#Max Leaf Nodes

plt.subplot(3,3,7)

feature_param = range(2,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_leaf_nodes=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Max Leaf Nodes')
forest = RandomForestClassifier(max_depth = 50,                                

                                min_samples_split =7,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 18)



parameter_grid = {'n_estimators' : [15, 100, 200],

                  'criterion' : ['gini', 'entropy'],

                  'max_features' : ['auto', 'sqrt', 'log2', None]

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
forest = RandomForestClassifier(n_estimators = 200,

                                criterion = 'entropy',

                                max_features = None)

parameter_grid = {

                  'max_depth' : [None, 50],

                  'min_samples_split' : [7, 11],

                  'min_weight_fraction_leaf' : [0.0, 0.2],

                  'max_leaf_nodes' : [18, 20],

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
#My optimal parameters

clf = RandomForestClassifier(n_estimators = 200,

                                criterion = 'entropy',

                                max_features = None,

                                max_depth = 50,                                

                                min_samples_split =7,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 18)



clf.fit(X, Y_train)

Y_pred_RF = clf.predict(Xt)



clf.score(X_test,y_test)