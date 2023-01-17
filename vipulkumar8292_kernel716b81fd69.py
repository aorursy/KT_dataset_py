import pandas as pd

import numpy as np
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

#reading train and test dataset
train.head()
test.head()
train.shape
test.shape
train.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
train.isnull().sum()
test.isnull().sum()
train.dropna(inplace=True)

test.dropna(inplace=True)
train.isnull().sum()
test.isnull().sum()
from sklearn.preprocessing import LabelEncoder
label_enc_train = LabelEncoder()

label_enc_test = LabelEncoder()
df_train = train[['Sex','Embarked']]

df_test = test[['Sex','Embarked']]
train[['Sex','Embarked']]= df_train.apply(label_enc_train.fit_transform)

test[['Sex','Embarked']] = df_test.apply(label_enc_test.fit_transform)
survived_train = np.array(train['Survived']).reshape(-1,1)
survived_train.astype(int)
y_train = survived_train

x_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

x_test = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(5,4),max_iter = 500,activation='relu')

mlp.fit(x_train,y_train)
mlp.score(x_train,y_train)
y_test = mlp.predict(x_test)

y_test
mlp.score(x_test,y_test)