# https://github.com/kaggle/docker-python

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, GridSearchCV

from subprocess import check_output
INPUT_PATH = '../input'

ENCODING = 'utf8'

TRAIN_PATH = INPUT_PATH + '/train.csv'

TEST_PATH = INPUT_PATH + '/test.csv'
print(check_output(["ls", INPUT_PATH]).decode(ENCODING))



# Results written to the current directory are saved as output
train = pd.read_csv(TRAIN_PATH)

train.head()
train.describe()
train.shape
def map_sex_to_number(data):

    sex = {'male': 0, 'female': 1}

    return data.replace({'Sex': sex})
train = map_sex_to_number(train)
def one_hot(data):

    one_hot = pd.get_dummies(data['Pclass'])

    one_hot = one_hot.add_prefix('passenger_class_')

    data = data.drop('Pclass', axis=1)

    data = data.join(one_hot)

    

    one_hot = pd.get_dummies(data['Embarked'])

    one_hot = one_hot.add_prefix('embarkation_')

    data = data.drop('Embarked', axis=1)

    data = data.join(one_hot)

    

    return data
train = one_hot(train)
sns.heatmap(train.corr())
train["Age"].fillna(train["Age"].mean(), inplace=True)
train['Cabin'].astype(str).str[0].drop_duplicates()
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
train_x = train.loc[:, 'Sex':]

train_x = preprocessing.scale(train_x)

train_x
train_y = train.loc[:, 'Survived']

train_y.head()
classifier = RandomForestClassifier()

cross_val_score(classifier, train_x, train_y, cv=10).mean()
classifier = GradientBoostingClassifier()

cross_val_score(classifier, train_x, train_y, cv=10).mean()
classifier = SVC()

cross_val_score(classifier, train_x, train_y, cv=10).mean()
classifier = MLPClassifier()

cross_val_score(classifier, train_x, train_y, cv=10).mean()
test = pd.read_csv(TEST_PATH)

identifier = test['PassengerId']

test = map_sex_to_number(test)

test = one_hot(test)

test["Age"].fillna(test["Age"].mean(), inplace=True)

passenger_class_3_mean_fare = test.loc[test['passenger_class_3'] == 1]['Fare'].mean()

test['Fare'].fillna(passenger_class_3_mean_fare, inplace=True)

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test = preprocessing.scale(test)
classifier = SVC()

classifier = classifier.fit(train_x, train_y)

survival = classifier.predict(test)
submission = pd.DataFrame(identifier)

survival = pd.DataFrame(survival)

submission = submission.join(survival)

submission.columns = ['PassengerId', 'Survived']

submission.to_csv('../working/submission.csv', index=False)