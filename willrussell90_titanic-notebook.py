# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data and view a summary

raw_data = pd.read_csv('../input/train.csv')

print(raw_data.columns)

raw_data.head()



#raw_data[raw_data.Age.isnull()].groupby('Survived').count()

print(raw_data.groupby(['Pclass', 'Sex', 'Embarked'])['Survived'].mean())

print(raw_data.Cabin.nunique())

print(raw_data.isnull().sum())

#raw_data.Cabin.isnull().mean()
def cabinMap(elem):

    d = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'N': 8}

    elem = d[elem[0]]

    return elem



#testing code for reformatting Cabin data



#df  = raw_data.copy()

#df['Cabin'] = df['Cabin'].fillna('N')

#df['Cabin'] = df['Cabin'].apply(cabinMap)

#df.head()
# define independent and dependent variables



y = raw_data.Survived

features = ['Pclass', 'Sex', 'Embarked']

X = raw_data[features].copy()

#X.isnull().sum()

#X.groupby('Age')['Age'].count()

X.groupby('Sex').count()
#reformat data to change sex into binary

def reformat(x):

    d_sex = {'female': 0, 'male': 1}

    d_emb = {'C': 0, 'Q': 1, 'S': 2, '3':3}

    x['Sex'] = [d_sex[sex] for sex in x['Sex'].values]

    x['Embarked'] = x['Embarked'].fillna('3')

    x['Embarked'] = [d_emb[elem] for elem in x['Embarked'].values]

    try:

        x['Cabin'] = x['Cabin'].fillna('N')

        x['Cabin'] = x['Cabin'].apply(cabinMap)

    except KeyError:

        pass



reformat(X)

#print(X.Embarked.isnull().sum(), X.Cabin.isnull().sum())

X.head()
# split training data into train and test subsets



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#If using Age then impute to add missing ages



#my_imputer = SimpleImputer()

#imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))

#imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))



# Fill in the lines below: imputation removed column names; put them back

#imputed_X_train.columns = train_X.columns

#imputed_X_valid.columns = val_X.columns



#print(train_X.Age.mean(), imputed_X_train.Age.mean())
scaler = StandardScaler()



train = scaler.fit_transform(train_X)

valid = scaler.transform(val_X)





print(train.shape)

print(valid.shape)
model = RandomForestClassifier(random_state=1)

model.fit(train,train_y)



# Get validation predictions and MAE

preds_val = model.predict(valid)

print("Accuracy:")

print(sum(val_y == preds_val)/len(val_y))
cross_val_score(model , train , train_y , cv=5)
def applyModel(X_test):

    X = X_test[features].copy()

    reformat(X)

    X = scaler.transform(X)

    preds = model.predict(X)

    return preds
X_test = pd.read_csv('../input/test.csv')

preds = applyModel(X_test)

output = pd.DataFrame({'PassengerId': X_test.PassengerId,

                       'Survived': preds})

output.to_csv('submission.csv', index=False)