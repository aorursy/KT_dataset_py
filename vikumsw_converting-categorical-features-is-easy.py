import numpy as np 

import pandas as pd 

import os

data = pd.read_csv("../input/train.csv")

data.drop(columns = ['Name','Ticket','Fare','Cabin','SibSp','Parch','Age','PassengerId','Survived','Pclass'], inplace=True)

data['Embarked'].fillna('S',inplace=True)

data.head(10)
print('Unique Values of Columns')

print('\tSex \t\t: ',data.Sex.unique())

print('\tEmbarked \t: ',data.Embarked.unique())
columnsToEncode = ['Sex','Embarked']

One_Hot_encoded = pd.get_dummies(data,columns= columnsToEncode)

One_Hot_encoded.head()
data.head()
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()



data_for_Label_Encoding = data.copy()

data_for_Label_Encoding['Sex'] = le.fit_transform(data_for_Label_Encoding[['Sex']])

data_for_Label_Encoding['Embarked'] = le.fit_transform(data_for_Label_Encoding[['Embarked']])
data_for_Label_Encoding.head()
data_for_Label_Encoding = data.copy()

data_for_Label_Encoding['Sex'].replace(['male','female'],[1,2],inplace=True)

data_for_Label_Encoding['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

data_for_Label_Encoding.head()