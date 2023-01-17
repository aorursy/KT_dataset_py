# Premise

import numpy as np

np.random.seed = 1

import pandas as pd

from IPython import embed

from sklearn.neighbors import KNeighborsClassifier





train_array = ['Sex', 'Pclass',

'Age', 'SibSp', 'Parch',

'Fare', 'Embarked']



target_index = np.arange(start=892, stop=1310, step=1)



x_train = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = train_array)

t_train = pd.read_csv("/kaggle/input/titanic/train.csv", usecols = ['Survived'])

x_test = pd.read_csv("/kaggle/input/titanic/test.csv", usecols = train_array)
# NaN detection

print("x_train:\n", x_train.isnull().any())

print("\nt_train:\n", t_train.isnull().any())

print("\nx_test:\n", x_test.isnull().any())
# a. Looking further into the sex and embarked columns

print(np.unique(x_test.Sex))

u, counts = np.unique(x_test.Embarked, return_counts=True)

print("Embarked:\n", u,"\nCounts:\n", counts)
# b. Further investigation into the Fare column of x_test:

u, counts = np.unique(x_test.Fare, return_counts = True)

print("Unique values:\n", u, "\nCounts:\n", counts)
print("x_train:\n", x_train.dtypes)

print("\n\nx_test:\n", x_test.dtypes)
print(x_test.describe())

print(x_train.describe())
# Further investigation into Age column with decimal points

print(x_test[(x_test['Age'] != x_test['Age'].round()) & ~(x_test.Age.isnull())])
# Cleansing Process



## Sex column: adding dummy numbers

x_train.Sex[x_train.Sex=='male'] = 1

x_train.Sex[x_train.Sex=='female'] = 2



x_test.Sex[x_test.Sex=='male'] = 1

x_test.Sex[x_test.Sex=='female'] = 2



## Embarked column: eliminating NaNs for the x_train, adding categorical numbers for both x_train and x_test 

x_train.Embarked[x_train.Embarked.isnull()] = 0

x_train.Embarked[x_train.Embarked=='Q'] = 1

x_train.Embarked[x_train.Embarked=='C'] = 2

x_train.Embarked[x_train.Embarked=='S'] = 3



x_test.Embarked[x_test.Embarked=='Q'] = 1

x_test.Embarked[x_test.Embarked=='C'] = 2

x_test.Embarked[x_test.Embarked=='S'] = 3





## Age column: adding 0 to NaNs

x_test.Age[x_test.Age.isnull()] = 0

x_train.Age[x_train.Age.isnull()] = 0





## Fare column: turning NaN into 0 for simplicity for the x_test

x_test.Fare[x_test.Fare.isnull()] = 0





## Data types conversion into float64

for key in train_array:

    x_train[key] = x_train[key].astype('float64')

    x_test[key] = x_test[key].astype('float64')

#To make sure that all the datum have been arranged properly



print("[NaN:]\nx_train:\n", x_train.isnull().any(), "\n\nx_test:\n", x_test.isnull().any(), \

     "\n\n[Data types:]\nx_train:\n", x_train.dtypes, "\n\nx_test:\n", x_test.dtypes)
#  KNN method



model = KNeighborsClassifier(n_neighbors=2)

model.fit(x_train, t_train)

predicted=model.predict(x_test)





output = pd.DataFrame({'PassengerId': target_index, 'Survived': predicted})

output.to_csv('Submission_yosher_mar27_v3.csv', index=False)

print("Your submission was successfully saved!")