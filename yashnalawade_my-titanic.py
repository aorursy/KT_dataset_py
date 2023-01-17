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

        

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



data = [train_data, test_data]

for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)



embarked_mode = train_data['Embarked'].mode()



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)



for dataset in data:

    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['Relatives'] > 0, 'travelled_alone'] = 'No'

    dataset.loc[dataset['Relatives'] == 0, 'travelled_alone'] = 'Yes'

    

train_data=data[0]

test_data=data[1]



y = train_data["Survived"]



features = ["Pclass", "Sex", "travelled_alone","Age","Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = LogisticRegression(max_iter=200)

model.fit(X, y)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")