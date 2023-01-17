import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])

df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())



df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
df_train = pd.get_dummies(df_train)

df_test = pd.get_dummies(df_test)
df_train.head()
import h2o

from h2o.automl import H2OAutoML

h2o.init()
htrain = h2o.H2OFrame(df_train)

htest = h2o.H2OFrame(df_test)



x = htrain.columns

y = 'Survived'

x.remove(y)



htrain[y] = htrain[y].asfactor()
aml = H2OAutoML(max_runtime_secs = 30000,nfolds=10)

aml.train(x=x, y =y, training_frame=htrain)

lb = aml.leaderboard



print (lb)

print('Generate predictions…')

test_y = aml.leader.predict(htest)

test_y = test_y.as_data_frame()
hpred = h2o.H2OFrame(test_y)

result = htest.cbind(hpred)
result = result[:, ["PassengerId", "predict"]]
submission  = result.as_data_frame()
submission.rename(index=str, columns={"predict": "Survived"}, inplace=True)

submission.head()
submission.to_csv('submission.csv', index=False)