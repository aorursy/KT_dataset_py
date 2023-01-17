import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore", "Mean of empty slice")
!conda install -y gxx_linux-64 gcc_linux-64 swig

!curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install    
!pip install auto-sklearn
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')                      

y = df_train.Survived

df_train.drop('Survived', axis=1, inplace=True)

n_train = df_train.PassengerId.count()

n_test = df_test.PassengerId.count()

df_todos = pd.concat([df_train, df_test])
df_todos['Family'] = df_todos['Name'].map(lambda x : x[0:x.find(',')])

df_todos['Ship_side'] = df_todos['Cabin'].str[0]
df_todos['Pclass'] = pd.factorize(df_todos['Pclass'])[0]

df_todos['Sex'] = pd.factorize(df_todos['Sex'])[0]

df_todos['Cabin'] = pd.factorize(df_todos['Cabin'])[0]

df_todos['Embarked'] = pd.factorize(df_todos['Embarked'])[0]

df_todos['Ticket'] = pd.factorize(df_todos['Ticket'])[0]

df_todos['Ship_side'] = pd.factorize(df_todos['Ship_side'])[0]

df_todos['Family'] = pd.factorize(df_todos['Family'])[0]

df_todos.loc[df_todos['Embarked'].isnull(), 'Embarked'] = 'S'

df_todos.loc[df_todos['Age'].isnull(), 'Age'] = int(df_todos.Age.median())

df_todos.loc[df_todos['Fare'].isnull(), 'Fare'] = int(df_todos.Fare.median())

df_todos.drop('Name', axis=1, inplace=True)
passenger_Id = df_todos['PassengerId']

X_train = df_todos[:n_train].values

X_test = df_todos[n_train:].values

y_train = y.values

passenger_Id_test = passenger_Id[n_train:].values
import autosklearn.classification

cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, ensemble_size=50,

                                                      resampling_strategy='cv', 

                                                       resampling_strategy_arguments={'folds':3})

cls.fit(X_train, y_train)

cls.refit(X_train, y_train)

cls.show_models()
preds_test = cls.predict(X_test)
df_result = pd.DataFrame(passenger_Id_test, columns=['PassengerId'])

df_result['Survived'] = (preds_test.astype('int'))

df_result.to_csv('submittion.csv', index=False)

df_result.head(3)
cls.show_models()