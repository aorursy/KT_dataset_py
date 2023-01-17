# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/titanic/train.csv")    ## Train dataset



df_test = pd.read_csv('/kaggle/input/titanic/test.csv')  ## Test dataset
df.head()
df_test.head()
df.shape, df_test.shape
df.isnull().sum()
df_test.isnull().sum()
mean_val = df['Age'].mean()

mean_val
df['Age'] = df['Age'].fillna(value=mean_val)
mode_val = df['Embarked'].mode()

mode_val
df['Embarked'] = df['Embarked'].fillna(value='S')  
df['Cabin'] = df['Cabin'].fillna(value='NaN')
df.isnull().sum()
df_test.isnull().sum()
df_test['Age'].mean()
df_test['Age'] = df_test['Age'].fillna(value=(df_test['Age'].mean()))
df_test['Cabin'].mode()
df_test['Cabin'] = df_test['Cabin'].fillna(value=(df_test['Cabin'].mode()[0]))
df_test['Fare'] = df_test['Fare'].fillna(value=(df_test['Fare'].mean()))
df_test.isnull().sum()
df = df.astype({'SibSp':'object','Parch':'object','Pclass':'object'})
df_test = df_test.astype({'SibSp':'object','Parch':'object','Pclass':'object'})
df = pd.get_dummies(df.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1))
df.head()
df.shape
df_test.head()
df_test['Parch'].value_counts()  ## train Data does notcontain Parch_9

## But test_X data contains Parch_9 its shape when we create dummies will (418,25), but we want it to be (418,24)

## It is very much imp that the no. of columns train_X and test_X match

## Now when we split the data to be trained we get shape of train_X with 24 columns 

## And so as test_X will be having 24 columns when we put Parch_9 into Parch_0
df_test['Parch'].replace({9:0},inplace=True)

df_test['Parch'].value_counts()
df_test = df_test.astype({'Parch':'object'})
test_X = pd.get_dummies(df_test.drop(['PassengerId','Name','Ticket','Cabin','Parch'],axis=1))

test_X.head()
test_X.shape
train_X = df.drop(['Survived'], axis=1)    ## X has the feature variables(Independent Var) which help in prediction of target

train_y = df['Survived']                  ## Y has the Dependent var



train_X.shape, train_y.shape
## MinMaxScaler scales down values in the range 0 to 1



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(train_X)
## X_scaled is an array O/P, convert it into Panda DataFrame

## X earlier contained the Features therefore it shld only now hold the scaled features



train_X = pd.DataFrame(X_scaled, columns=train_X.columns)
train_X.head()   ## Range = 0 to 1
testx_scaled = scaler.fit_transform(test_X)



test_X = pd.DataFrame(testx_scaled, columns=test_X.columns)

test_X.head()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=12)
rf.fit(train_X, train_y)
rf.score(train_X, train_y)
train_predict = rf.predict(train_X)
test_pred = rf.predict(test_X)
test_pred
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': test_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")