# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Load in the train data set
train0 = pd.read_csv('../input/train.csv')
train0.head(10)
# ... and test dataset
test0 = pd.read_csv('../input/test.csv')
test0.head(10)
# So we have these features in train/test dataframes:
# * PassengerId : Passenger ID
# * Pclass : Passenger Class (1, 2, 3)
# * Name
# * Sex : (male, female)
# * Age
# * SibSp : Number of Siblings/Spouses Aboard
# * Parch : Number of Parents/Children Aboard
# * Ticket
# * Fare
# * Cabin : (many of which are nan)
# * Embarked :  (C = Cherbourg; Q = Queenstown; S = Southampton)
# And the label in train dataframe:
# * Survived : (0, 1)

# The explanation of the values can be found from: http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf

# Start feature engineering
# Most of the works are from: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

dfs = [train0, test0]

# Fix the nan values

for df in dfs:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else ''

# Add some composite features

for df in dfs:
    df['NameLength'] = df['Name'].apply(len)
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
for df in dfs:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
for df in dfs:
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
# Mapping

for df in dfs:
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    df.loc[ df['Fare'] <= 7.91, 'Fare']                          = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare']                             = 3
    df['Fare'] = df['Fare'].astype(int)
    df.loc[ df['Age'] <= 16, 'Age']                     = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']                      = 4
    df['Age'] = df['Age'].fillna(df['Age'].median())

train0.head(10)

# Keep only interested features

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train0.drop(drop_elements, axis = 1)
test = test0.drop(drop_elements, axis = 1)

train.head(10)
test.head(10)
# Look for the columns with higher corr:
train.corr().nlargest(11, 'Survived')['Survived']
# It seems all of them are legit.

# Build the linear regressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_x = train.iloc[:, [1, 11]].values
train_y = train.iloc[:, [0]].values
linear = LinearRegression()
linear.fit(train_x, train_y)
print(linear.intercept_)
print(linear.coef_)
# See how the regressor prediction does with the train dataframe

prediction = [(1 if x > 0.5 else 0) for x in linear.predict(train_x)]
mean_squared_error(prediction, train_y)
# Now predict on the test data
test_x = test.iloc[:, [0, 10]].values
test_predict = [(1 if x > 0.5 else 0) for x in linear.predict(test_x)]
output = pd.concat([test0['PassengerId'], pd.DataFrame(test_predict)], axis=1)
output.columns = ['PassengerId', 'Survived_Prediction']
output.describe()
output.to_csv('assignment2_linear.csv', index=False)
# Build random forest regressor

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=300, random_state=0)
forest.fit(train_x, train_y)
# Score the regressor
forest.score(train_x, train_y)
# Make prediction with forest regressor

test_predict = [(1 if x > 0.5 else 0) for x in forest.predict(test_x)]
output = pd.concat([test0['PassengerId'], pd.DataFrame(test_predict)], axis=1)
output.columns = ['PassengerId', 'Survived_Prediction']
output.describe()
output.to_csv('assignment2_forest.csv', index=False)
# NN

import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers

model = Sequential()

# Add layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training
model.fit(train_x, train_y, batch_size=32, epochs=200)
# NN prediction

test_predict = [(1 if x > 0.5 else 0) for x in model.predict(test_x)]
output = pd.concat([test0['PassengerId'], pd.DataFrame(test_predict)], axis=1)
output.columns = ['PassengerId', 'Survived_Prediction']
output.describe()
output.to_csv('assignment2_nn.csv', index=False)