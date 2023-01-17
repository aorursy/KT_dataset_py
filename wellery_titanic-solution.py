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
# Importing libraries



import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
# Some magic for reading Data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Read train and test data

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



# Show first 5 lines

train_data.head()
# Divide train data for Features and Result

train_y = train_data["Survived"]

train_X = train_data.drop("Survived", axis=1)



# Just copying test data

test_X = test_data.copy()



# Show first 5 lines

train_X.head()
drop_columns = ["Name","Ticket"]

train_X = train_X.drop(drop_columns, axis=1)

test_X = test_X.drop(drop_columns, axis=1)



train_X.head()
# Missings in Train data

print("Training data:")

print(train_X.shape)



missing_val_count_by_column = (train_X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



#Missings in Test data

print("Test data:")

print(test_X.shape)



missing_val_count_by_column = (test_X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# drop Cabin column

drop_columns = ["Cabin"]

train_X = train_X.drop(drop_columns, axis=1)

test_X = test_X.drop(drop_columns, axis=1)



# Make new columns indicating what will be imputed

col = "Age"

train_X[col + '_was_missing'] = train_X[col].isnull()

test_X[col + '_was_missing'] = test_X[col].isnull()



# Imputation

train_X["Age"].fillna(train_X["Age"].mean(), inplace=True)

for col in ["Age","Fare"]:

    test_X[col].fillna(test_X[col].mean(), inplace=True)



train_X.head()
# Get list of categorical variables

s = (train_X.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
train_X = pd.get_dummies(train_X, columns=['Sex','Embarked'])

test_X = pd.get_dummies(test_X, columns=['Sex','Embarked'])

train_X.head()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(train_X, train_y)

predictions = model.predict(test_X)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")