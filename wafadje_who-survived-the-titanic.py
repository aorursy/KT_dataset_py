import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

all_data = [train, test]
pp.ProfileReport(train)

print( train[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )
print( train[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
for data in all_data:

    data['family_size'] = data['SibSp'] + data['Parch'] + 1

print( train[["family_size","Survived"]].groupby(["family_size"], as_index = False).mean() )
for data in all_data:

    data['is_alone'] = 0

    data.loc[data['family_size'] == 1, 'is_alone'] = 1

print (train[['is_alone', 'Survived']].groupby(['is_alone'], as_index=False).mean())
for data in all_data:

    data['Embarked'] = data['Embarked'].fillna('S')

print( train[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )

for data in all_data:

    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

train['category_fare'] = pd.qcut(train_data['Fare'], 3)

print( train[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )
for data in all_data:

    age_avg  = data['Age'].mean()

    age_std  = data['Age'].std()

    age_null = data['Age'].isnull().sum()



    random_list = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null)

    data['Age'][np.isnan(data['Age'])] = random_list

    data['Age'] = data['Age'].astype(int)



train['category_age'] = pd.cut(train_data['Age'], 5)

print( train[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )

for data in all_data:

    data['Situation'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    data['Situation'] = data['Situation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

    data['Situation'] = data['Situation'].replace('Mlle','Miss')

    data['Situation'] = data['Situation'].replace('Ms','Miss')

    data['Situation'] = data['Situation'].replace('Mme','Mrs')

    

print(pd.crosstab(train['Situation'], train['Sex']))

print("----------------------")

print(train[['Situation','Survived']].groupby(['Situation'], as_index = False).mean())
for data in all_data:



    #Mapping Sex

    sex_map = { 'female':0 , 'male':1 }

    data['Sex'] = data['Sex'].map(sex_map).astype(int)



    #Mapping Title

    title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

    data['Situation'] = data['Situation'].map(title_map)

    data['Situation'] = data['Situation'].fillna(0)



    #Mapping Embarked

    embark_map = {'S':0, 'C':1, 'Q':2}

    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)



    #Mapping Fare

    data.loc[ data['Fare'] <= 8.662, 'Fare']                            = 0

    data.loc[(data['Fare'] > 8.662) & (data['Fare'] <= 26.0), 'Fare'] = 1

    data.loc[ data['Fare'] > 26.0, 'Fare']                               = 2

    data['Fare'] = data['Fare'].astype(int)



    #Mapping Age

    data.loc[ data['Age'] <= 26.947, 'Age']                       = 0

    data.loc[(data['Age'] > 26.947) & (data['Age'] <= 53.473), 'Age'] = 1

    data.loc[ data['Age'] > 53.473, 'Age']                        = 2

#Feature Selection

#Create list of columns to drop

drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "family_size"]



#Drop columns from both data sets

train_data = train.drop(drop_elements, axis = 1)

train_data = train.drop(['PassengerId','category_fare', 'category_age'], axis = 1)

test_data = test.drop(drop_elements, axis = 1)



#Print ready to use data

print(train_data.head(10))

print(test_data)
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
#Create a CSV with results

submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": Y_pred

})

submission.to_csv('submission.csv', index = False)