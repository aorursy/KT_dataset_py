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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
train.describe(include='all')
train.isna().sum()
train.columns
import matplotlib.pyplot as plt

class_wise = train[['Pclass', 'Survived']].groupby('Pclass')['Survived'].mean()

print(class_wise)

class_wise.plot(kind='bar')

plt.show()
sex_wise = train[['Sex', 'Survived']].groupby('Sex')['Survived'].mean()

print(sex_wise)

sex_wise.plot(kind='bar')

plt.show()
sibsp_wise = train[['SibSp', 'Survived']].groupby('SibSp')['Survived'].mean()

print(sibsp_wise)

sibsp_wise.plot(kind='bar')

plt.show()
parch_wise = train[['Parch', 'Survived']].groupby('Parch')['Survived'].mean()

print(parch_wise)

parch_wise.plot(kind='bar')

plt.show()
fare_wise = train[['Fare', 'Survived']]

fare_wise['Fare'] = pd.cut(fare_wise['Fare'],10)

fare_wise.groupby('Fare')['Survived'].mean().plot(kind='bar')

plt.show()
embark_wise = train[['Embarked', 'Survived']].groupby('Embarked')['Survived'].mean()

print(embark_wise)

embark_wise.plot(kind='bar')

plt.show()
cabin_wise_data  =  train[['Cabin', 'Survived']]

cabin_wise_data['Cabin2'] = [x[0] if x==x else 'blank' for x in cabin_wise_data['Cabin']]

cabin_wise = cabin_wise_data[['Cabin2', 'Survived']].groupby('Cabin2')['Survived'].mean()

print(cabin_wise)

cabin_wise.plot(kind='bar')

plt.show()
cabin_wise_fare  =  train[['Cabin', 'Fare']]

cabin_wise_fare['Cabin2'] = [x[0] if x==x else 'blank' for x in cabin_wise_fare['Cabin']]

cabinvsfare = cabin_wise_fare[['Cabin2', 'Fare']].groupby('Cabin2')['Fare'].mean()

print(cabinvsfare)

cabinvsfare.plot(kind='bar')

plt.show()
test.isna().sum()
#combine data for preprocessing 

data = [train.copy(deep=True),test.copy(deep=True)]

for df in data:

    print(df.isna().sum())

    print('-'*20)
for df in data:

    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['Fare'].fillna(df['Fare'].median(),inplace=True)

    df['Cabin2'] = [x[0] if x==x else 'N' for x in df['Cabin']]
for df in data:

    print(df.isna().sum())

    print('-'*20)
for df in data:

    df.drop(['Ticket'], axis =1, inplace=True)

    df.drop(['Cabin'], axis=1,inplace=True)

for df in data:

    print(df.isna().sum())

    print('-'*20)
for df in data:

    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand = True)[0]

    title_counts = df['Title'].value_counts()

    df['Title'] = df['Title'].apply(lambda x: "Rare" if title_counts.loc[x]<10 else x)

    print(df['Title'].value_counts())

    try:

        print(df.groupby('Title')['Survived'].agg({'Count': 'size', 'Avg': 'mean'}).sort_values(by='Count'))

    except:

        continue
for df in data: 

    df.drop('Name', axis = 1, inplace =True)

    df.drop('PassengerId', axis=1, inplace=True)
for df in data:

    df.info()
y = data[0]['Survived'].copy(deep=True)

X = data[0].drop('Survived', axis=1).copy(deep=True)

X_submission = data[1].copy(deep = True)

print(X.columns,X_submission.columns)
y = y.astype('category')

y
X_dummy = pd.get_dummies(X)

X_submission_dummy = pd.get_dummies(X_submission)

X_submission_dummy.loc[:,'Cabin2_T'] = 0

X_submission_dummy = X_submission_dummy.loc[:,X_dummy.columns]
print(X_dummy.columns)

print('-'*100)

print(X_submission_dummy.columns)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



clf = DecisionTreeClassifier(random_state=1)

cross_val_score(clf, X_dummy, y, cv=10)

clf.fit(X_dummy, y)

output = clf.predict(X_submission_dummy)

output
test['Survived'] = output

submission = test[['PassengerId','Survived']]

submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(max_depth = 2, n_estimators = 200, random_state=1)

clf.fit(X_dummy,y)

test['Survived'] = clf.predict(X_submission_dummy)

submission = test[['PassengerId','Survived']]

submission.to_csv('submission2.csv', index=False)
"""

from sklearn.model_selection import GridSearchCV

parameters = {"loss":["deviance"],

              "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

              "max_depth":[1,2,3],

              "max_features":["log2","sqrt"],

              "criterion": ["friedman_mse",  "mae"],

              "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

              "n_estimators":[100,200]}



clf = GridSearchCV(GradientBoostingClassifier(), parameters, n_jobs = -1,cv=10)

clf.fit(X_dummy,y)

best_model = clf.best_estimator_

test['Survived'] = best_model.predict(X_submission_dummy)

submission = test[['PassengerId','Survived']]

submission.to_csv('submission3.csv', index=False)

"""
from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

from keras.callbacks import EarlyStopping



labelencoder_y_1 = LabelEncoder()

y = pd.get_dummies(y)

shape = (X_dummy.shape)[1]



def get_new_model(shape):

    model = Sequential()

    model.add(Dense(200,activation = 'sigmoid', input_shape = (shape,)))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model

early_stopping_monitor = EarlyStopping(patience=3)

model = get_new_model(shape)

model.fit(X_dummy,y, validation_split=0.20,epochs = 30, callbacks=[early_stopping_monitor])

test['Survived'] = [int(x) for x in np.rint(model.predict(X_submission_dummy))[:,1]]

submission = test[['PassengerId','Survived']]

submission.to_csv('submission4.csv', index=False)