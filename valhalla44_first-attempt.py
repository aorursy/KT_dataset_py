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
import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.shape
train_df.isna().sum()
train_df.describe()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
def rank_attib(attrib):

    return train_df[[attrib, 'Survived']].groupby([attrib], as_index=False).mean().sort_values(by='Survived', ascending=False)
rank_attib("Sex").plot(kind='bar', x='Sex', y='Survived')
rank_attib('SibSp')
rank_attib('Parch')
rank_attib('Embarked')
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_df.head()
train_df[["PassengerId", "Cabin"]].groupby("Cabin").count().sort_values(by='PassengerId', ascending=False).head()
_new = train_df["Cabin"].fillna("X00")

_new = _new.str.split()

_new = _new.apply(lambda x: [x[0][0], x[0][1:]])
train_df.head()

_new.head()
train_df[["Deck", "Room"]] = pd.DataFrame(_new.values.tolist())
train_df.head()
train_df = train_df.drop("Cabin", axis=1)
titles = train_df['Name'].str.extract('([\w]*\.)')

train_df['Title'] = titles
train_df.head()
train_df[["PassengerId","Deck"]].groupby('Deck').count().sort_values(by='PassengerId', ascending=False)
rank_attib('Deck').plot(kind='bar', x='Deck', y='Survived')
rank_attib("Title")
train_df.info()
train_df['Title'] = train_df['Title'].astype('category')

train_df['Deck'] = train_df['Deck'].astype('category')

train_df['Room'] = pd.to_numeric(train_df['Room'])
import numpy as np

train_df['Room'] = train_df['Room'].replace(np.nan, 0)

train_df['Room'] = train_df['Room'].astype('int')
train_df.info()
train_df['Room'] = train_df['Room'].astype('int16')

#train_df['Age'] = train_df['Age'].astype('int8')

train_df['Survived'] = train_df['Survived'].astype('int8')

train_df['Pclass'] = train_df['Pclass'].astype('int8')

train_df['SibSp'] = train_df['SibSp'].astype('int8')

train_df['Parch'] = train_df['Parch'].astype('int8')

train_df['Sex'] = train_df['Sex'].astype('category')
train_df.info()
train_df.describe()
train_df.at[351, "Room"] = None

train_df.at[351, "Room"] = 128
train_df.iloc[351, train_df.columns.get_loc('Room')]
train_df.plot.scatter(x="Room", y="Survived")
train_df['Room'] = train_df['Room'].astype('int16')
train_df.plot.scatter(x="Room", y="Survived")
train_df.plot.scatter(x="Fare", y="Survived")
train_df['Embarked'].unique()
train_df['Embarked'] = train_df['Embarked'].astype('category')
train_df = train_df.drop('Ticket', axis=1)
train_df = train_df.drop('Name', axis=1)
train_target = train_df['Survived']

train_features = train_df.drop('Survived', axis=1)

train_features['Embarked'] = train_features['Embarked'].cat.codes

train_features['Deck'] = train_features['Deck'].cat.codes

train_features['Sex'] = train_features['Sex'].cat.codes

train_features['Title'] = train_features['Title'].cat.codes
train_features['Age'] = train_features['Age'].fillna(train_df['Age'].mean())

train_features['Embarked'] = train_features['Embarked'].fillna('N')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_features, train_target)
test_df = pd.read_csv('../input/test.csv')

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



_new = test_df["Cabin"].fillna("X00")

_new = _new.str.split()

_new = _new.apply(lambda x: [x[0][0], x[0][1:]])



test_df[["Deck", "Room"]] = pd.DataFrame(_new.values.tolist())

test_df = test_df.drop("Cabin", axis=1)



titles = test_df['Name'].str.extract('([\w]*\.)')

test_df['Title'] = titles

test_df['Title'] = test_df['Title'].replace(['Mlle.', 'Mme.', 'Lady.', 'Countess.','Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')

test_df['Title'] = test_df['Title'].astype('category')



#####

test_df['Deck'] = test_df['Deck'].astype('category')

test_df['Room'] = pd.to_numeric(test_df['Room'])

test_df['Room'] = test_df['Room'].replace(np.nan, 0)

test_df['Room'] = test_df['Room'].astype('int')

test_df['Room'] = test_df['Room'].astype('int16')

#test_df['Age'] = test_df['Age'].astype('int8')

#test_df['Survived'] = test_df['Survived'].astype('int8')

test_df['Pclass'] = test_df['Pclass'].astype('int8')

test_df['SibSp'] = test_df['SibSp'].astype('int8')

test_df['Parch'] = test_df['Parch'].astype('int8')

test_df['Sex'] = test_df['Sex'].astype('category')

test_df['Embarked'] = test_df['Embarked'].astype('category')

test_df = test_df.drop('Ticket', axis=1)

test_df = test_df.drop('Name', axis=1)

##

#test_target = test_df['Survived']

test_features = test_df

test_features['Embarked'] = test_features['Embarked'].cat.codes

test_features['Deck'] = test_features['Deck'].cat.codes

test_features['Sex'] = test_features['Sex'].cat.codes

test_features['Title'] = test_features['Title'].cat.codes

test_features['Age'] = test_features['Age'].fillna(test_df['Age'].mean())

test_features['Fare'] = test_features['Fare'].fillna(test_df['Fare'].mean())

test_features['Embarked'] = test_features['Embarked'].fillna('N')
test_features.info()
predictions = knn.predict(test_features)
result = pd.read_csv('../input/gender_submission.csv')
result['Predicted'] = predictions
result.head()
result['accurate'] = result['Predicted'] == result['Survived']
result['accurate'].groupby(result['accurate']).count()
sub = pd.DataFrame({'PassengerId':result['PassengerId'], 'Survived':result['Predicted']})
sub.head()

sub.to_csv('../submission.csv', index=False)