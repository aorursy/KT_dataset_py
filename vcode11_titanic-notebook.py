# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# Basic Checking of shapes

print('Number of Training Examples {}'.format(train_df.shape))

print('Number of Test Examples {}'.format(test_df.shape))

print('Train Features:\n', train_df.columns)

print('Test Features\n', test_df.columns)
train_df.info()
test_df.info()
def concat_df(train_data, test_data):

    return pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

def divide_df(merged_df):

    return merged_df.loc[:890], merged_df.loc[891:].drop(['Survived'], axis=1)



df = concat_df(train_df, test_df)
df.corr().abs()
age_by_pclass_sex = df.groupby(['Sex', 'Pclass'])[['Age']].apply(lambda x: x.sum())

age_by_pclass_sex
df['Age'] = df.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df[df['Embarked'].isna()]
df['Embarked'] = df['Embarked'].fillna('S') # Filling Southampton as this is the value for martha evelyn
df[df['Fare'].isna()]
fare_for_alone_traveller_of_3rd_class = df.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3,0,0]

df['Fare'] = df['Fare'].fillna(fare_for_alone_traveller_of_3rd_class)
cabin_decks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'M']

def map_to_deck(cabin: str) -> str:

    for deck in cabin_decks:

        if deck in cabin:

            return deck

    return cabin

df['Cabin'] = df['Cabin'].fillna('M')

df['Cabin'] = df['Cabin'].apply(map_to_deck)

df['Cabin'] = df['Cabin'].replace('T', 'A')

survival_by_deck = {}

for deck, survived in zip(df['Cabin'], df['Survived']):

    if deck == "Missing":

        continue

    if np.isnan(survived):

        continue

    if deck not in survival_by_deck:

        survival_by_deck[deck] = [0,0]

    survival_by_deck[deck][int(survived)]+=1

for k, v in survival_by_deck.items():

    survival_by_deck[k] = v[1]/(v[0]+v[1])

sns.barplot(x=list(survival_by_deck.keys()), y=list(survival_by_deck.values()))
df.groupby(['Pclass', 'Cabin']).size()
df['Cabin'] = df['Cabin'].replace(['A', 'B', 'C'], 'ABC')

df['Cabin'] = df['Cabin'].replace(['D', 'E'], 'DE')

df['Cabin'] = df['Cabin'].replace(['F', 'G'], 'FG')

df['Cabin'].value_counts()
df.isna().sum()
train_df, test_df = divide_df(df)
survived_stats = df['Survived'].value_counts().reset_index()

plt.figure(figsize=(8,6))

sns.barplot(x=survived_stats['index'], y=survived_stats['Survived'])

plt.title('Survival Percentage')

total = survived_stats['Survived'].sum()

plt.xlabel('')

plt.xticks((0,1), ['Not Survived {:.2f}%'.format(survived_stats.loc[0,'Survived']/total), 'Survived {:.2f}%'.format(survived_stats.loc[1,'Survived']/total) ])
plt.figure(figsize=(10,10))



plt.subplot(1,2,1)

plt.title('Train set correlations')

sns.heatmap(train_df.corr(), annot=True, linewidth=0.5, cmap='coolwarm')



plt.subplot(1,2,2)

plt.title('Test set correlations')

sns.heatmap(test_df.corr(), annot=True, linewidth=0.5, cmap='coolwarm')

plt.tight_layout()
plt.figure(figsize=(16,10))

plt.subplot(1,2,1)

sns.distplot(a=train_df[train_df['Survived'] == 1]['Age'], label='Survived')

sns.distplot(a=train_df[train_df['Survived'] == 0]['Age'], label='Not Survived')

plt.title('Distribution of Age and Survival')

plt.legend()



plt.subplot(1,2,2)

sns.distplot(a=train_df['Age'], label='Train Set')

sns.distplot(a=test_df['Age'], label='Test Set')

plt.title('Ages Test set vs train set')

plt.legend()
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

sns.distplot(a=train_df[train_df['Survived'] == 1]['Fare'], label='Survived')

sns.distplot(a=train_df[train_df['Survived'] == 0]['Fare'], label='Not Survived')

plt.title('Distribution of Fare and Survival')

plt.legend()



plt.subplot(1,2,2)

sns.distplot(a=train_df['Fare'], label='Train Set')

sns.distplot(a=test_df['Fare'], label='Test Set')

plt.title('Fares Test set vs train set')

plt.legend()
def plot_data(categoryA):

    data = train_df.groupby([categoryA, 'Survived']).size().reset_index()

    data.rename(columns={0:'Count'}, inplace=True)

    sns.barplot(x=categoryA, y='Count', hue='Survived', data=data)

    plt.title('{} vs Survival'.format(categoryA))

    

# embarked_vs_survival = train_df.groupby(['Embarked', 'Survived']).size().reset_index()

# embarked_vs_survival.rename(columns={0:'Count'}, inplace=True)

plt.figure(figsize=(20,10))

plt.subplot(2,3,1)

plot_data('Embarked')



plt.subplot(2,3,2)

plot_data('Sex')



plt.subplot(2,3,3)

plot_data('Pclass')



plt.subplot(2,3,4)

plot_data('SibSp')



plt.subplot(2,3,5)

plot_data('Parch')



plt.subplot(2,3,6)

plot_data('Cabin')



plt.tight_layout()

titles = ['Mr', 'Mrs', 'Ms','Master', 'Dr','Miss', 'Don', 'Capt', 'Col', 'Dona', 'Rev', 'Mlle', 'Mme', 'Major', 'Jonkheer', 'Countess']

def to_title(name: str) -> str:

    for title in titles:

        if title in name:

            return title

    return name

def replace_titles(x: pd.DataFrame) -> str:

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Army/Clergy/Doctor'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

df['Title'] = df['Name'].apply(to_title)

df['Title'] = df.apply(replace_titles, axis=1)

df.head(10)


title_df = df.groupby(['Title', 'Survived']).size().reset_index()

plt.figure(figsize=(14,8))

sns.barplot(x='Title', y=0, hue='Survived', data=title_df)
df['Family_size'] = df['SibSp'] + df['Parch']

family_size_df = df.groupby(['Family_size', 'Survived']).size()

plt.figure(figsize=(16,10))

sns.barplot(x='Family_size', y=0, hue='Survived', data=family_size_df.reset_index())

percentages = []

for i in range(11):

    try:

        percentage = family_size_df.loc[(i,1.0)]/family_size_df.loc[i].sum()

        percentages.append(percentage)

    except:

        percentage = 0

        percentages.append(percentage)

labels = ['Size {} \nSurvived {:.2f}%'.format(i, percentages[i]) for i in range(11)]

plt.xticks(tuple(range(11)), labels)

family_size_df.head(25)
df.head()
df['AgeClass'] = df['Age']*df['Pclass']

df.head()
from sklearn.preprocessing import LabelEncoder



le_cols = ['Sex', 'Embarked','Title','Cabin']

for col in le_cols:

    df[col] = LabelEncoder().fit_transform(df[col])

cols_to_drop = ['Name', 'Ticket']

df.head()

df = df.drop(cols_to_drop, axis=1)

train_df, test_df = divide_df(df)

train_df.head()
test_df.head()
X = train_df.drop(['Survived','PassengerId'], axis=1)

y = train_df.Survived

print(X.shape, y.shape)

X.head()
import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import GridSearchCV



cat_cols = ['Embarked', 'Title', 'Sex', 'Pclass', 'Cabin',]

ord_cols = ['Family_size',  'SibSp', 'Parch']

num_cols = ['AgeClass','Age', 'Fare',]

preprocessing = ColumnTransformer(transformers=[

    ('cat_cols', OneHotEncoder(handle_unknown='ignore'), cat_cols),

    ('ord_cols',SimpleImputer(), ord_cols ),

     ('numerical_cols', StandardScaler(), num_cols),

])

pipeline = Pipeline(steps=[

    ('preprocessing', preprocessing),

    ('model', RandomForestClassifier(random_state=0, n_jobs=-1)),

])

param_grid = {

        'model__max_depth': [5, 10, 15],

        'model__min_samples_split': [10, 20, 30],

        'model__n_estimators': [100, 200, 300],

        'model__min_samples_leaf': [5, 10, 15],

        "model__bootstrap": [True],

        "model__criterion": ["entropy"]

}



clf = GridSearchCV(estimator=pipeline,param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=10)

start = time.time()

clf.fit(X,y)

print('Took {}s'.format(time.time()-start))

print('Best Score: {} ,\n Param: {}'.format(clf.best_score_, clf.best_params_))
# estimator = clf.best_estimator_
scores = cross_val_score(estimator=clf.best_estimator_, X=X, y=y, n_jobs=-1, scoring='accuracy', cv=10)

print(scores.mean(), scores.std())

X.head()
test_df.head()
from sklearn.metrics import accuracy_score

pipeline = clf.best_estimator_

pipeline.fit(X,y)

print(accuracy_score(y, pipeline.predict(X)))
X.head()
test_df.head()
X_test = test_df.drop(['PassengerId'], axis=1)

X_test.head()
prediction = np.array(pipeline.predict(X_test), dtype=np.int)

output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':prediction})

output.head()
output.to_csv('submission.csv', index=False)