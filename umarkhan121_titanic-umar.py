import numpy as np 

import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
def data_cleaning(data):

    data.drop(['Cabin'], axis = 1, inplace = True)

    data.Age = data.Age.fillna(data.Age.median())

    data.Embarked=data.Embarked.fillna(data.Embarked.mode())

#     data.dropna(inplace = True)

    feature=["PassengerId","Name", 'Ticket']

    data_out=data[[x for x in data.columns if x not in feature]]

    ohe=pd.get_dummies(data_out)

    scaler=MinMaxScaler()

    std_data=pd.DataFrame(columns = ohe.columns, data = scaler.fit_transform(ohe))

    return std_data

    

    
missin=["NaN"]



train=pd.read_csv("/kaggle/input/titanic/train.csv",na_values=missin)

y=train["Survived"]

train=data_cleaning(train)

print(train.isna().sum().sort_values())

# print(train.describe(include="all"))



train.drop(["Survived"],inplace=True,axis=1)

print(train,y)
# train
# train.isnull().sum()

# train.drop(['Cabin'], axis = 1, inplace = True)
# train.Age = train.Age.fillna(train.Age.median())
# train.dropna(inplace = True)
# print(train.describe(include="all"))

# y=train["Survived"]

# train.drop(["Survived"],inplace=True,axis=1)

# print(train,y)

# feature=["PassengerId","Name", 'Ticket']

# x_train=train[[x for x in train.columns if x not in feature]]

# x_train
# x=data_cleaning(train)

# x
# ohe=pd.get_dummies(x_train)

# scaler=MinMaxScaler()

# std_data=pd.DataFrame(columns = ohe.columns, data = scaler.fit_transform(ohe))
# std_data
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
testin=data_cleaning(test_data)

print(test_data.isna().sum())
testin.Fare=testin.Fare.fillna(testin.Fare.median())

testin
clf=MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation='relu', verbose = True, max_iter=1000)
clf.fit(train, y)
std_predictions = clf.predict(testin)
rf=RandomForestClassifier(n_estimators=100, max_depth=7)
rf.fit(train,y)
predictions=rf.predict(testin)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': std_predictions})

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")