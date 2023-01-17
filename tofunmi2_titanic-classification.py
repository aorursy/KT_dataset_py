import os

os.getcwd()
os.chdir('/kaggle/input')

os.listdir()
import pandas as pd #for structuring the data

import numpy as np #for mathematical manipulation of the data

import sklearn #for preprocessing and model building

import matplotlib.pyplot as plt #for visualization

import seaborn as sns #for visualization
train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')
train_data.info()
train_data.head()
train_data.describe()
dataTrain = train_data.drop(['passenger_ID', 'name', 'ticket','cabin'], axis = 1)

passengerid = test_data['passenger_ID']

dataTest= test_data.drop(['passenger_ID','name','ticket', 'cabin'], axis = 1)
dataTrain.head()
dataTest.head()
dataTrain['pclass'].value_counts()
sns.countplot(x = 'pclass', hue = 'survived', data = dataTrain)
dataTrain['sex'].value_counts()

sns.countplot(x = 'sex', hue = 'survived', data = dataTrain)
dataTrain = pd.get_dummies(dataTrain, columns = ['sex']) #one hot encode the train data

dataTrain.head()
dataTest = pd.get_dummies(dataTest, columns = ['sex']) #one-hot encode the test data

sns.boxplot(x = 'age', orient = 'horizontal', data = dataTrain)
agemean, agemedian, agemode = dataTrain['age'].mean(),dataTrain['age'].median(),dataTrain['age'].mode()[0]

print(agemean, agemedian, agemode)
dataTrain['age'].fillna(agemedian, inplace = True)
dataTest['age'].fillna(agemedian, inplace = True)
dataTrain.info()
sns.boxplot(x = 'age', orient = 'horizontal', data = dataTrain)
def transform(x):

    if x<5:

        out = 5

    elif x>50:

        out = 50

    else:

        out = x

    return out



dataTrain['age'] = dataTrain['age'].apply(transform)

dataTest['age'] = dataTest['age'].apply(transform)
sns.boxplot(x = 'age', orient = 'horizontal', data = dataTrain)
sns.distplot(dataTrain['age'])
dataTrain['sibsp'].value_counts()
sns.countplot(x = 'sibsp', hue = 'survived', data = dataTrain)
def transform_sib(x):

    if x in [2.0,3.0,4.0,5.0,8.0]:

        out = 1.0

    else:

        out = x

    return out



dataTrain['sibsp'] = dataTrain['sibsp'].apply(transform_sib)

dataTest['sibsp'] = dataTest['sibsp'].apply(transform_sib)
sns.countplot(x = 'sibsp', hue = 'survived', data = dataTrain)
dataTrain['parch'].value_counts()
sns.countplot(x = 'parch', hue = 'survived', data = dataTrain)
def transform_parch(x):

  if x in [2.0,3.0,4.0,5.0,6.0,9.0]:

    out = 1.0

  else:

    out = x

  return out



dataTrain['parch'] = dataTrain['parch'].apply(transform_parch)

dataTest['parch'] = dataTest['parch'].apply(transform_parch)
sns.countplot(x = 'parch', hue = 'survived', data = dataTrain)
sns.distplot(dataTrain['fare'])
dataTrain['embarked'].value_counts()
sns.countplot(x = 'embarked', hue = 'survived', data = dataTrain)
dataTrain = pd.get_dummies(dataTrain, columns = ['embarked']) #one hot encode the train data

dataTest = pd.get_dummies(dataTest, columns = ['embarked']) #one hot encode the test data
dataTrain.head()
dataTest.head()
y = dataTrain['survived']

x = dataTrain.drop('survived', axis =1)

x.shape, y.shape
# Normalize/ standardize the data

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

stdscale = MinMaxScaler()

x_new = stdscale.fit_transform(x)

testd = stdscale.transform(dataTest)

x_new.shape, testd.shape
X = pd.DataFrame(x_new, columns = x.columns)

testData = pd.DataFrame(testd, columns = dataTest.columns)

X.head()
testData.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

x_train.shape, y_train.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score





lr = LogisticRegression()

rand = RandomForestClassifier()

gbr = GradientBoostingClassifier()

x_train.describe()
np.unique(y_train)
for i in  x_train.columns:

  x_train[i].fillna(x_train[i].median(), inplace = True)



for i in  x_test.columns:

  x_test[i].fillna(x_test[i].median(), inplace = True)



for i in  testData.columns:

  testData[i].fillna(testData[i].median(), inplace = True)
lr.fit(x_train, y_train)

lr.score(x_train, y_train)
pred = lr.predict(x_test)

print('accuracy_score for logistic is: ', accuracy_score(y_test, pred))

print('f1_score for logistic is: ', f1_score(y_test, pred))

print('precision for logistic is : ', precision_score(y_test, pred))

print('recall score for logistic is: ', recall_score(y_test, pred))

# prediction on test data

final_prediction = lr.predict(testData)



dict_data = {}

dict_data['passenger_ID'] = passengerid

dict_data['survived'] = final_prediction

frame = pd.DataFrame(dict_data)

#frame.to_csv('submission_log.csv', index = False)
rand = RandomForestClassifier()

rand.fit(x_train, y_train)

rand.score(x_train, y_train)
pred = rand.predict(x_test)

print('accuracy_score for random forest is: ', accuracy_score(y_test, pred))

print('f1_score for random forest is: ', f1_score(y_test, pred))

print('precision for random forest is : ', precision_score(y_test, pred))

print('recall score for random forest is: ', recall_score(y_test, pred))

# prediction on test data

final_prediction = rand.predict(testData)



dict_data = {}

dict_data['passenger_ID'] = passengerid

dict_data['survived'] = final_prediction

frame = pd.DataFrame(dict_data)

#frame.to_csv('rand_submission.csv', index = False)
gbr = GradientBoostingClassifier()

gbr.fit(x_train, y_train)

gbr.score(x_train, y_train)
pred = gbr.predict(x_test)

print('accuracy_score for gradient boost is: ', accuracy_score(y_test, pred))

print('f1_score for gradient boost is: ', f1_score(y_test, pred))

print('precision for gradient boost is : ', precision_score(y_test, pred))

print('recall score for gradient boost is: ', recall_score(y_test, pred))

# prediction on test data

final_prediction = gbr.predict(testData)



dict_data = {}

dict_data['passenger_ID'] = passengerid

dict_data['survived'] = final_prediction

frame = pd.DataFrame(dict_data)

#frame.to_csv('gbr_submission.csv', index = False)