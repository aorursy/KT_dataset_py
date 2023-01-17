import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

np.random.seed(7)
# Input data files are available in the "../input/" directory.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# We do the same Missing value treatment as before

# Replacing the missing values

# train - Age, Cabin, Embarked

# test - Age, Fare, Cabin



# 1. Replace the Age in Train

tr_avage = train.Age.mean()

tr_sdage = train.Age.std()

tr_misage = train.Age.isnull().sum()

rand_age = np.random.randint(tr_avage - tr_sdage, tr_avage + tr_sdage, size=tr_misage)

train['Age'][np.isnan(train['Age'])] = rand_age

train['Age'] = train['Age'].astype(int)



# 2. Replace the Age in Test

te_avage = test.Age.mean()

te_sdage = test.Age.std()

te_misage = test.Age.isnull().sum()

rand_age = np.random.randint(te_avage - te_sdage, te_avage + te_sdage, size=te_misage)

test['Age'][np.isnan(test['Age'])] = rand_age

test['Age'] = test['Age'].astype(int)
# 3. Replace the Embarked in Train

# Distribution of Embarked in train S-644, C-168, Q-77

train['Embarked'] = train['Embarked'].fillna('S')



# 4. Treat the cabin for both test and train as a new varibale "Is_Cabin"

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# 5. Replace the Fare in test with a median value

med =  test.Fare.median()

test['Fare'] =  test['Fare'].fillna(med)
# Create new Features - 1. FamilySize 2. Solo traveller 3. Age bucket



# 1. FamilySize

train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + test['Parch']



# 2. Create New Feature Solo Traveller

train['Solo'] = train['FamilySize'].apply(lambda x: 0 if x>0 else 1)

test['Solo'] = test['FamilySize'].apply(lambda x: 0 if x>0 else 1)



# For Train

train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)



def Age(row):

    if row['Age'] < 16:

        return 'VY'

    elif row['Age'] < 32:

        return 'Y'

    elif row['Age'] < 48:

        return 'M'

    elif row['Age'] < 64:

        return 'O'

    else:

        return 'VO'

    

train['CategoricalAge'] = train.apply(lambda row: Age(row), axis=1)

test['CategoricalAge'] = test.apply(lambda row: Age(row), axis=1)
# Final Feature Selection Droping the ones which may look not necessary

drop_list = ['PassengerId', 'Name', 'Cabin', 'Ticket', 'Age']

ftrain = train.drop(drop_list, axis = 1)

ftest = test.drop(drop_list, axis = 1)
# labelling the Dataset before passing to a model

# 1. Map the variable Sex

ftrain['Sex'] = ftrain['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

ftest['Sex'] = ftest['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# 2. Map the variable Embarked

ftrain['Embarked'] = ftrain['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

ftest['Embarked'] = ftest['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# 3. Map the Categorical Age

ftrain['CategoricalAge'] = ftrain['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)

ftest['CategoricalAge'] = ftest['CategoricalAge'].map( {'VY': 0, 'Y': 1, 'M': 2, 'O': 3, 'VO': 4} ).astype(int)
# Creating the X and Y for both Train and Test

y_train = ftrain['Survived'].ravel()

ftrain = ftrain.drop(['Survived'], axis=1)

x_train = ftrain.values # Creates an array of the train data

x_test = ftest.values # Creats an array of the test data

print(x_train.shape)
# Creating the Deep Learning Model

model = Sequential()

model.add(Dense(20, input_dim=10, activation='relu', kernel_initializer="uniform"))

model.add(Dense(12, activation='relu', kernel_initializer="uniform"))

model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=10,  verbose=2)
pred3 = model.predict(x_test)

rounded = [int(round(x[0])) for x in pred3]
fpred = pd.DataFrame({'out': rounded})


final_sub3 = pd.DataFrame({ 'PassengerId': test.PassengerId, 'Survived': fpred.out }) 

final_sub3.to_csv("Sub5.csv", index=False)