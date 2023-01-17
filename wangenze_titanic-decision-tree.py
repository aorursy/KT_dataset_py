import pandas as pd

import numpy as np
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.info()
data_x = data_train[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]

data_y = data_train['Survived']

age_mean = data_x['Age'].dropna().median()

fare_mean = data_x['Fare'].dropna().median()

data_x['Age'].fillna(age_mean, inplace=True)

data_x['Fare'].fillna(fare_mean, inplace=True)

data_x['Embarked'].fillna('S', inplace=True)

for i in range(1, 4):

    data_x.loc[data_x.Pclass == i, 'Pclass'] = str(i)

data_x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=33)

y_train.value_counts()
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=False)

x_train = vec.fit_transform(x_train.to_dict(orient='record'))

x_test = vec.transform(x_test.to_dict(orient='record'))

vec.get_feature_names()
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

dtc_y_predict = dtc.predict(x_test)
from sklearn.metrics import classification_report
dtc.score(x_test, y_test)
print(classification_report(y_test, dtc_y_predict, target_names=['died', 'surived']))
run_x = data_test[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]

run_x['Age'].fillna(age_mean, inplace=True)

run_x['Fare'].fillna(fare_mean, inplace=True)

for i in range(1, 4):

    run_x.loc[run_x.Pclass == i, 'Pclass'] = str(i)

run_x = vec.transform(run_x.to_dict(orient='record'))

run_y_predict = dtc.predict(run_x)
pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': run_y_predict}).to_csv('gender_submission_1.csv', index =False)