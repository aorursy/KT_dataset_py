#import sklearn.grid_search as sklg



import sklearn.model_selection as sklm



import sys

print(sys.version)
# Import libraries



import numpy as np

import pandas as pd



import subprocess as sp



import sklearn as skl
# Load data



print(sp.check_output(['ls', '../input']).decode('utf8'))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print('Train DS shape: ', str(train.shape))

print('Test DS shape: ' + str(test.shape))
# Show all data



train
# Review data



def reviewData(data):

    

    d = data.describe()

    

    print('\nColumns:')

    print("All: " + str(data.columns))

    print("Numeric: " + str(d.columns))

    print("Non-null: ")

    print(data.count())

    

    print('\nDescribe:')

    print(d)

    

    return





reviewData(train)
# Print column values



def printColumnValues(data, columns):

    

    data = data[columns]

    

    for c in data.columns:

        print('\nColumn ' + c)

        print(data[c].value_counts())

        

    return





printColumnValues(train, ['Sex', 'Cabin', 'Embarked'])
# Preprocess data



def preprocessData(data):

    

    data = data.copy()

    

    # Sex

    data['Sex_'] = 0

    data.set_value(data['Sex'] == 'male', 'Sex_', 1)

    data.set_value(data['Sex'] == 'female', 'Sex_', -1)

    

    # Embarked

    data['Embarked_S'] = 0

    data['Embarked_C'] = 0

    data['Embarked_Q'] = 0

    data.set_value(data['Embarked'] == 'S', 'Embarked_S', 1)

    data.set_value(data['Embarked'] == 'C', 'Embarked_C', 1)

    data.set_value(data['Embarked'] == 'Q', 'Embarked_Q', 1)

    

    # 

    data['Age_'] = data['Age']

    data.set_value(data['Age_'].isnull(), 'Age_', 28.0)

    

    data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

    

    return data



train_dat = preprocessData(train)



reviewData(train_dat)

printColumnValues(train_dat, ['Sex_', 'Embarked_S', 'Embarked_C', 'Embarked_Q'])
train = preprocessData(train)

reviewData(train)