import pandas as pd

import re

import h2o

from h2o.automl import H2OAutoML
h2o.init()
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
def get_title(name):

    title = re.search(' ([A-Za-z]+)\.', name)

    if title:

        return title.group(1)

    return '0'



def process_data(df):

    #preprocessing based https://www.kaggle.com/paulorzp/titanic-gp-model-training

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['Alone'] = 0

    df.loc[df['FamilySize'] == 1, 'Alone'] = 1

    #df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

    #df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    

    df['Cabin'].fillna('0', inplace=True)

    df['Cabin'] = df['Cabin'].str[0]

    #df['Cabin'] = df['Cabin'].map({'0':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8})

    #df['Cabin'] = df['Cabin'].astype(int)

    



    #title_mapping = {'0':0, 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

    title_replace = {'0':'0', 'Mlle': 'Rare', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Rare', 'Rev': 'Mr',

                     'Don': 'Mr', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Mrs',

                     'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}

    df['Title'] = df['Name'].map(get_title)

    df.replace({'Title': title_replace}, inplace=True)

    #df['Title'] = df['Title'].map(title_mapping)

    df['Title'].fillna('0', inplace=True)

    #df['Title'] = df['Title'].astype(int)

    

    df.fillna(0, inplace=True)

    return h2o.H2OFrame(df)
train = process_data(train)

test = process_data(test)
train
X_train, y_test = train.split_frame(ratios=[.80], seed=2020)
y = "Survived"

x = train.columns

x.remove(y)
X_train[y] = X_train[y].asfactor()

y_test[y] = y_test[y].asfactor()
model = H2OAutoML(max_models=20, nfolds=10,  max_runtime_secs=300, max_runtime_secs_per_model=90, verbosity='info')
model.train(x = x, y = y, training_frame = X_train, leaderboard_frame = y_test)
model.leaderboard


model.leader
pred = model.leader.predict(test)
submission = test['PassengerId'].as_data_frame()

submission['Survived'] = pred['predict'].as_data_frame()

submission.to_csv('submission.csv', index=False)