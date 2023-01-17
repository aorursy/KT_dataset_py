# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# stores to dataframe

import pandas as pd

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df
df.info()

# there are missing values and object(string)
df.isnull().sum()

# show count of missing value
# fill median to missing values of Age

df["Age"].fillna(df.Age.median(),inplace = True)
# remove Cabin

df = df.drop("Cabin", axis =1)
# number of appearances for each values in Embarked

df['Embarked'].value_counts()
# fill mode value to missing values of Embarked

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) 
df.info()

# there are no missing values.

# there are object(string)
# remove PassengerId, Name and Ticket

df = df.drop("PassengerId", axis =1)

df = df.drop("Name", axis =1)

df = df.drop("Ticket", axis =1)

df.head()
# label-encode to Sex

df['Sex'].replace(['male', 'female'],[0,1], inplace =True)

df.head()
# label-encode to Embarked

df['Embarked'].replace(['C','Q','S'],[0,1,2], inplace =True)

df.head()
df.info()

# there are no missing values and no object(string)
# split test data set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    df.iloc[:, 1:], df.iloc[:, 0], test_size = 0.20, random_state=1)

print('X_train shape: ',X_train.shape,' y_train shape: ', y_train.shape,' X_test shape: ', X_test.shape,' y_test shape: ', y_test.shape)
from sklearn.ensemble import RandomForestClassifier



# create random forest classifier

model = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators = 1000,max_depth=None,max_features=2)



# train model

model.fit(X_train, y_train)
predict_y = model.predict(X_test)

predict_y
import numpy as np

np.array(y_test)
# culclate accuracy

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predict_y)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
test_passenger_id = test_data.iloc[:, :1]

test_passenger_id
test_data.info()

# there are missing values and object(string)
test_data.isnull().sum()

# show count of missing value
# fill median to missing values of Age

test_data["Age"].fillna(test_data.Age.median(),inplace = True)



# remove Cabin

test_data = test_data.drop("Cabin", axis =1)



# fill median to missing values of Age

test_data["Fare"].fillna(test_data.Fare.median(),inplace = True)



# fill mode value to missing values of Embarked

test_data['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) 
test_data.info()

# there are no missing values.

# there are object(string)
# remove PassengerId, Name and Ticket

test_data = test_data.drop("PassengerId", axis =1)

test_data = test_data.drop("Name", axis =1)

test_data = test_data.drop("Ticket", axis =1)

test_data.head()
# label-encode to Sex

test_data['Sex'].replace(['male', 'female'],[0,1], inplace =True)

test_data.head()
# label-encode to Embarked

test_data['Embarked'].replace(['C','Q','S'],[0,1,2], inplace =True)

test_data.head()
test_data.info()

# there are no missing values and no object(string)
predictions = model.predict(test_data.values)

predictions
output = pd.DataFrame({'PassengerId': test_passenger_id.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")