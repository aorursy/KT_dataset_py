import numpy as np

import pandas as pd
!ls ../input/titanic
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=False)
data.head()
print(len(train), len(test), len(data))
data.isnull().sum()
data['Pclass'].value_counts()
#data['Sex'].replace(['male','female'], [0,1], inplace=True)

data['Sex'].replace(['male','female'], [0, 1], inplace=True)
data.head()
data['Embarked'].value_counts()
data['Embarked'].fillna(('S'), inplace=True)

#data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Embarked'].replace(['S','C','Q'], [0,1,2], inplace=True)
data.head()
## np.mean() 平均を求める
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data.head()
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data.head()
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred[:20]
sub = gender_submission

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index=False)