# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv', index_col = "PassengerId")

train.head()

test = pd.read_csv('../input/test.csv', index_col = "PassengerId")

test.head()
sns.countplot(data = train, x = 'Sex', hue = 'Survived')
pd.pivot_table(train, index = 'Sex', values = 'Survived')    
sns.countplot(data = train, x = 'Pclass', hue = 'Survived')
pd.pivot_table(train, index = 'Pclass', values = 'Survived')
sns.countplot(data = train, x = 'Embarked', hue = 'Survived')
pd.pivot_table(train, index = 'Embarked', values = 'Survived')
sns.lmplot(data = train, x = 'Age',  y='Fare', hue = 'Survived', fit_reg = False)
low_Fare = train[train['Fare'] < 100]
sns.lmplot(data = low_Fare, x = 'Age', y = 'Fare', hue = 'Survived', fit_reg = False)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train[['SibSp', 'Parch', 'FamilySize']].head()
sns.countplot(data = train, x = 'FamilySize', hue = 'Survived')
train.loc[train['FamilySize'] == 1, 'FamilyType'] = 'Single'

train.loc[(train['FamilySize'] > 1) & (train['FamilySize'] < 5), 'FamilyType'] = 'Nuclear'

train.loc[train['FamilySize'] >= 5, 'FamilyType'] = 'Big'



train[['FamilySize', 'FamilyType']].head()
sns.countplot(data = train, x = 'FamilyType', hue = 'Survived')
pd.pivot_table(train, index = 'FamilyType', values = 'Survived')
train['Name'].head()
def get_title(name):

    return name.split(', ')[1].split('. ')[0]



train['Name'].apply(get_title).unique()
train.loc[train['Name'].str.contains('Mr'), 'Title'] = 'Mr'

train.loc[train['Name'].str.contains('Mrs'), 'Title'] = 'Mrs'

train.loc[train['Name'].str.contains('Miss'), 'Title'] = 'Miss'

train.loc[train['Name'].str.contains('Master'), 'Title'] = 'Master'



train[['Name','Title']].head()
sns.countplot(data = train, x = 'Title', hue = 'Survived')
pd.pivot_table(train, index = "Title", values = 'Survived')
train.loc[train['Sex'] == 'female', 'Sex_encode'] = 0 

train.loc[train['Sex'] == 'male', 'Sex_encode'] = 1

test.loc[test['Sex'] == 'female', 'Sex_encode'] = 0 

test.loc[test['Sex'] == 'male', 'Sex_encode'] = 1
train[['Sex', 'Sex_encode']].head()
test[['Sex', 'Sex_encode']].head()
train[train['Fare'].isnull()]
test[test['Fare'].isnull()]
train['Fillin_Fare'] = train['Fare']
test['Fillin_Fare'] = test['Fare']
test.loc[test['Fare'].isnull(), 'Fillin_Fare']= 0

test.loc[test['Fare'].isnull(), ['Fare', 'Fillin_Fare']]
train['Embarked_C'] = train['Embarked'] == 'C'

train['Embarked_Q'] = train['Embarked'] == 'Q'

train['Embarked_S'] = train['Embarked'] == 'S'



train[['Embarked','Embarked_S', 'Embarked_Q', 'Embarked_C']].head()
test['Embarked_C'] = test['Embarked'] == 'C'

test['Embarked_Q'] = test['Embarked'] == 'Q'

test['Embarked_S'] = test['Embarked'] == 'S'



test[['Embarked','Embarked_S', 'Embarked_Q', 'Embarked_C']].head()
train['Child'] = train['Age'] < 15

train[['Age', 'Child']].head()
test['Child'] = test['Age'] < 15

test[['Age', 'Child']].head()
train['Single'] = train['FamilySize'] == 1

train['Nuclear'] = (train['FamilySize'] > 1)  & (train['FamilySize'] < 5)

train['Big'] = train['FamilySize'] >= 5



train[['FamilySize', 'Single', 'Nuclear', 'Big']].head()
test['Single'] = test['FamilySize'] == 1

test['Nuclear'] = (test['FamilySize'] > 1)  & (test['FamilySize'] < 5)

test['Big'] = test['FamilySize'] >= 5



test[['FamilySize', 'Single', 'Nuclear', 'Big']].head()
train['Master'] = train['Name'].str.contains('Master')



train[['Name', 'Master']].head()
test['Master'] = test['Name'].str.contains('Master')



test[['Name', 'Master']].head()
feature_names = ['Pclass', 'Sex_encode', 'Fillin_Fare', 'Embarked_C','Embarked_Q', 'Embarked_S', 'Child', "Single", "Nuclear", 'Big', 'Master']

feature_names
label_name = 'Survived'
X_train = train[feature_names]

X_train.head()
X_test = test[feature_names]

X_test.head()
y_train = train['Survived']

y_train.head()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(max_depth = 8, random_state = 0)

model
model.fit(X_train, y_train)
import graphviz

from sklearn.tree import export_graphviz



dot_tree = export_graphviz(model, feature_names = feature_names, class_names=["Perish", "Survived"], out_file = None)

graphviz.Source(dot_tree)
predictions = model.predict(X_test)

print(predictions.shape)

predictions[0:10]
submission = pd.read_csv('../input/gender_submission.csv', index_col = 'PassengerId')
submission['Survived'] = predictions

print(submission.shape)

submission.head()