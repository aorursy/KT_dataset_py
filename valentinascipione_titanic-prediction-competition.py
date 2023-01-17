# Import the Pandas library

import pandas as pd



# Load the train and test datasets to create two DataFrames

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#Print the `head` of the train dataframe

train.head()
train.describe()
train.shape
# Passengers that survived vs passengers that passed away, as proportions

train['Survived'].value_counts(normalize=True)
# Normalized male survival

print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True))
# Normalized female survival

print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True))
# Create the column Child and assign to 'NaN'

train["Child"] = float('NaN')



# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"] < 18] = 1

train["Child"][train["Age"] >= 18] = 0



# Print normalized Survival Rates for passengers under 18

train["Survived"][train["Child"] == 1].value_counts(normalize = True)
# Create a copy of test: test_one

test_one = test



# Initialize a Survived column to 0

test_one['Survived'] = 0



# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test_one['Survived'][test_one['Sex'] == 'female'] = 1

test_one[['PassengerId', 'Survived']].to_csv('my_solution.csv', index=False)