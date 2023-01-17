## some basic data anlysis, label encoder is very useful for quickly converting categorical data into valid inputs.



import numpy as np

import pandas as pd

import copy

data=pd.read_csv('../input/train.csv')

print(data.shape) 



data.head()

#print(data.columns)
data.isnull().sum()
import math

thirdClass = data[data["Pclass"] == 3];

print("number of 3rd class: " +str(len(thirdClass))) #so no, not all the NaNs are in third class., also weird that only 491 steerage passengers?... this is a subset of the data

print("number of NaNs in 3rd: %d" % thirdClass["Cabin"].isna().sum())



#set all third class cabins to '3rd'?



#check second class

secondClass = data[data["Pclass"] == 2];

print("number of 2nd class: " +str(len(secondClass))) #so no, not all the NaNs are in third class., also weird that only 491 steerage passengers?... this is a subset of the data

print("number of NaNs in 2nd: %d" % secondClass["Cabin"].isna().sum())

#print((len(data[data["Pclass"] == 2]), len(data[data["Pclass"] == 1])))



#so there are even some first class people without an assigned cabin..damn

data['Cabin']  = data.apply(

    lambda row: '3' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 3 else row['Cabin'],

    axis=1

)

data['Cabin']  = data.apply(

    lambda row: '2' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 2 else row['Cabin'],

    axis=1

)

data['Cabin']  = data.apply(

    lambda row: '1' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 1 else row['Cabin'],

    axis=1

)
col = list(data.columns);

print(col)

#col.remove('Cabin');

#col.remove('Age'); 

col.remove('Embarked');

col.remove('PassengerId'); #not useful

col.remove('Name')

print(col)

GoodData = data[col];



X = GoodData;

y = GoodData['Survived'].values;

X.drop('Survived', axis = 1, inplace = True);

print(X.shape)
## convert all categorical data to numeric categories for classification

from sklearn.preprocessing import LabelEncoder;

for i in range(X.shape[1]):

    if X.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(X.iloc[:,i].values))

        X.iloc[:,i] = lbl.transform(list(X.iloc[:,i].values))

print(X.head())

X.fillna(0, inplace = True);
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 1);

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

rf.fit(X_train,y_train);

print(rf.score(X_test,y_test));
## write predictions... so we need to process it just like the train set

print(test.head())
test = pd.read_csv('../input/test.csv')

testIDs = test['PassengerId']

testcol = col;

#testcol.remove('Survived')

test['Cabin']  = data.apply(

    lambda row: '3' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 3 else row['Cabin'],

    axis=1

)

test['Cabin']  = data.apply(

    lambda row: '2' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 2 else row['Cabin'],

    axis=1

)

test['Cabin']  = data.apply(

    lambda row: '1' if not isinstance(row['Cabin'], str) and math.isnan(row['Cabin']) and row['Pclass'] == 1 else row['Cabin'],

    axis=1

)

Xtest = test[col]

## convert all categorical data to numeric categories for classification

for i in range(Xtest.shape[1]):

    if Xtest.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(Xtest.iloc[:,i].values))

        Xtest.iloc[:,i] = lbl.transform(list(Xtest.iloc[:,i].values))

print(Xtest.head())

Xtest.fillna(0, inplace = True);



ypred = rf.predict(Xtest)

submission = pd.DataFrame(ypred)

submission['PassengerId'] = testIDs



submission.to_csv('submission.csv', index=False)