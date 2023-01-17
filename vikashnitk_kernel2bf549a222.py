# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.head()
train_df = train_df.drop(['PassengerId'], axis=1)

train_df.head()
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)

train_df.head()
data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

train_df["Age"].isnull().sum()

train_df.head()
common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

train_df.head()
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df.head()
data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df.head()
genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)

train_df.head()
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

train_df.head()
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_test.head()
X_train.head()
Y_train.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_test
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam
# Model

model = Sequential()



# first input layer with first hidden layer in a single statement

model.add( Dense(12, input_shape=(9,), activation='relu') )

# 10 is the size(no. of neurons) of first hidden layer, 4 is the no. of features in the input layer

# input_shape=(7,)  can also be written as   input_dim=7



# second hiden layer

model.add(Dense(12,activation='relu')) 

 



# output layer

model.add(Dense(1,activation='sigmoid')) # 3 = no. of neurons in output layer as four categories of labels are there



# compile method receives three arguments: "an optimizer", "a loss function" and "a list of metrics"

model.compile(Adam(lr=0.04),'binary_crossentropy', ['accuracy'])
model.fit(X_train, Y_train, epochs=1000, batch_size=50)
predictions = model.predict(X_test)
predictions
new_predictions=[]

for x in predictions:

    if x[0] < 0.5:

        new_predictions.append(0)

    else:

        new_predictions.append(1)

   
new_predictions
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': new_predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")