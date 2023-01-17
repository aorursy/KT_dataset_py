

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

X_test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
print(train_data.info())

print('_'*50)

print(X_test.info())
print(train_data.isna().sum())

print('_'*50)

print(X_test.isna().sum())
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)

train_data['Cabin'].fillna('N0', inplace = True)

train_data['Deck'] = train_data["Cabin"].str.slice(0,1)

train_data['Embarked'].fillna('S', inplace = True)
sex_dummy = pd.get_dummies(train_data['Sex'], drop_first = True)

class_dummy = pd.get_dummies(train_data['Pclass'], drop_first = True)

embarked_dummy = pd.get_dummies(train_data['Embarked'], drop_first = True)

deck_dummy = pd.get_dummies(train_data['Deck'], drop_first = True)
train_data.drop(['Pclass','Name','Cabin','Deck', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

train_data = pd.concat([train_data, sex_dummy, class_dummy,embarked_dummy,deck_dummy], axis = 1)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

X_test['Cabin'].fillna('N0', inplace = True)

X_test['Deck'] = X_test["Cabin"].str.slice(0,1)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)
sex_dummy = pd.get_dummies(X_test['Sex'], drop_first = True)

class_dummy = pd.get_dummies(X_test['Pclass'], drop_first = True)

embarked_dummy = pd.get_dummies(X_test['Embarked'], drop_first = True)

deck_dummy = pd.get_dummies(X_test['Deck'], drop_first = True)
X_test.drop(['Pclass','Name','Cabin','Deck', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

X_test = pd.concat([X_test, sex_dummy, class_dummy,embarked_dummy,deck_dummy], axis = 1)
X_test['T'] = 0
print(train_data.columns)

print('_'*50)

print(X_test.columns)
y = train_data['Survived']

train_data = train_data.drop('Survived' , axis = 1)
X_train, X_val, y_train, y_val = train_test_split(train_data,y, test_size = 0.1, random_state = 0                                                 )
X_train.info()
classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)

print((cm[0,0]+cm[1,1])/cm.sum())
y_submit = classifier.predict(X_test)
results_df = pd.DataFrame()

results_df['PassengerId'] = X_test['PassengerId']

results_df["Survived"] = y_submit

results_df.to_csv("Predictions", index=False)