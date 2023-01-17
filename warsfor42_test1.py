import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

y_true = pd.read_csv("../input/genderclassmodel.csv")

# female = 0, Male = 1
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values
test.Embarked = test.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# Collect the test data's PassengerIds before dropping it
ids = train['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
# Collect the test data's PassengerIds before dropping it
ids = test['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

# All the ages with no data -> make the median of all Ages
median_age = train['Age'].dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age'] = median_age
# All the ages with no data -> make the median of all Ages
median_age = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.loc[ (test.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    # loop 0 to 2
    for f in range(0,3):                                              
        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()
    # loop 0 to 2
    for f in range(0,3):                                             
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
        
# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train.values
test_data = test.values


print ('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)

print (output)

from sklearn.metrics import accuracy_score

accuracy_score(y_true.Survived, output)
