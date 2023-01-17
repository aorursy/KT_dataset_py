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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
df = pd.concat([train.drop('Survived', axis=1), test])

df.reset_index(inplace=True, drop=True)

print(df.iloc[891:].shape)
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return



df['Title'] = df['Name'].apply(get_title)



def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title is 'Dr':

        if x['Sex'] is 'male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



df['Title'] = df.apply(replace_titles, axis=1)
df['Age'] = df.groupby(['Title', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))

df['AgeBin'] = pd.qcut(df['Age'], q=5, labels=range(5))



df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)



df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = 1 

df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

df['IsAlone'].value_counts()



df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform(np.mean), inplace=True)

df['FareBin'] = pd.qcut(df['Fare'], 5, labels=range(5))



cabins = df[['Cabin']].copy()

cabins['CabinData'] = cabins['Cabin'].isnull().apply(lambda x: not x)

cabins['Deck'] = cabins['Cabin'].str.slice(0, 1)

cabins['Room'] = cabins['Cabin'].str.slice(1, 5).str.extract('([0-9]+)', expand=False).astype('float')

cabins[cabins['CabinData']]

cabins[cabins['Deck'] == 'F']

cabins.drop(['Cabin', 'CabinData'], axis=1, inplace=True, errors='ignore')

cabins['Deck'] = cabins['Deck'].fillna('N')

cabins['Room'] = cabins['Room'].fillna(cabins['Room'].mean())



def one_hot_column(df, label, drop_col=False):

    one_hot = pd.get_dummies(df[label], prefix=label)

    if drop_col:

        df = df.drop(label, axis=1)

    df = df.join(one_hot)

    return df



def one_hot(df, labels, drop_col=False):

    for label in labels:

        df = one_hot_column(df, label, drop_col)

    return df



cabins = one_hot(cabins, ['Deck'], drop_col=True)



df['Cabin'] = np.argmax(cabins.drop('Room', axis=1).values, axis=1)

df[['Sex', 'Embarked', 'Title']] = df[['Sex', 'Embarked', 'Title']].astype('category')

df['Sex'] = df['Sex'].cat.codes

df['Embarked'] = df['Embarked'].cat.codes

df['Title'] = df['Title'].cat.codes



df['FareBin'] = df['FareBin'].astype(int)

df['Age'] = df['Age'].astype(int)

df['AgeBin'] = df['AgeBin'].astype(int)



df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True) # 'Cabin'



y = train['Survived'].copy()

X = df.iloc[:891].copy()

X_new = df.iloc[891:].copy()

print(X.shape, y.shape, X_new.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from gplearn.genetic import SymbolicRegressor

from sklearn.ensemble import RandomForestRegressor

from gplearn.functions import make_function
def _protected_division(x1, x2):

    with np.errstate(divide='ignore', invalid='ignore'):

        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

    

protected_division = make_function(function=_protected_division,

                                   name='protected_division', arity=2)



function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',

                'abs', 'neg', 'inv', 'max', 'min', 'cos', 'sin', 'tan', protected_division]



est_gp = SymbolicRegressor(population_size=2000,

                           generations=30, 

                           stopping_criteria=0.01,

                           p_crossover=0.7, 

                           function_set=function_set,

                           p_subtree_mutation=0.1,

                           p_hoist_mutation=0.05, 

                           p_point_mutation=0.1,

                           max_samples=0.9, 

                           verbose=1,

                           parsimony_coefficient=0.01, 

                           random_state=42)



est_gp.fit(X_train.values, y_train.values)

print(est_gp.score(X_train.values, y_train.values))

print(est_gp.score(X_test.values, y_test.values))

print(est_gp._program)
X_pred = est_gp.predict(X_new)

X_pred = X_pred.astype(np.int8)



submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                           'Survived': X_pred})



submission.to_csv('./submission-gp.csv', index=False)