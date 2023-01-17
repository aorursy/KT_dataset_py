import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.describe(include='all')
train.head()
train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
train = train.drop(['Embarked'], axis = 1)

test = test.drop(['Embarked'], axis = 1)
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
corr = train.corr()

fig, ax = plt.subplots(figsize=(9,6))

sns.heatmap(corr, annot=True, linewidths=1.5 , fmt='.2f',ax=ax)

plt.show()
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
print(pd.isnull(test).sum())
print(pd.isnull(train).sum())
train['Age'].fillna((train['Age'].mean()), inplace=True)

test['Age'].fillna((test['Age'].mean()), inplace=True)
train['Fare'].fillna((train['Fare'].mean()), inplace=True)

test['Fare'].fillna((test['Fare'].mean()), inplace=True)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

train[['Fare']] = scaler.fit_transform(train[['Fare']])

test[['Fare']] = scaler.fit_transform(test[['Fare']])
train[['Age']] = scaler.fit_transform(train[['Age']])

test[['Age']] = scaler.fit_transform(test[['Age']])
train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)