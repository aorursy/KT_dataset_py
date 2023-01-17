#3I0203 UI1
#importy funkcíí

import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import sys

!{sys.executable} -m pip install skorch

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer

from skorch import NeuralNetClassifier

from skorch.helper import predefined_split

from skorch.callbacks import EarlyStopping

from torch.utils.data import TensorDataset

import torch.nn as nn

import torch
#načítanie dát

train_df = pd.read_csv('../input/titanic/train.csv')

Xtest_df = pd.read_csv('../input/titanic/test.csv')
#výpis stĺpcov

print(train_df.columns.values)
#zahodenie mena, čísla lístku, kabíny a čísla pasažiera, podľa mňa nemali vplyv na prežitie

train_df = train_df.drop('Name', axis=1,)

train_df = train_df.drop('Ticket', axis=1,)

train_df = train_df.drop('Cabin', axis=1,)

train_df = train_df.drop('PassengerId', axis=1,)
#výpis nových stĺpcov

print(train_df.columns.values)
#výpis chývajúcich hodnôt

feat_list = list(train_df.columns.values)



for feat in feat_list:

    print (feat,": ",sum(pd.isnull(train_df[feat])))
#doplnienie prázdnych hodnôt

train_df["Embarked"].mode()

train_df["Embarked"] = train_df["Embarked"].fillna("S")

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
#overenie doplnenia

feat_list = list(train_df.columns.values)



for feat in feat_list:

    print (feat,": ",sum(pd.isnull(train_df[feat])))
#informatívny výpis

train_df.head()
#deľba dátovej množiny

df_train_valid, df_test = train_test_split(train_df, test_size=0.25,

                                     stratify=train_df['Survived'],

                                     random_state=4)

df_train, df_valid = train_test_split(df_train_valid, test_size=0.05/0.75,

                                     stratify=df_train_valid['Survived'],

                                     random_state=4)
#rozdelenie stĺpcov a predspracovanie, pipeline

categorical_inputs = ["Sex","Embarked"]

numeric_inputs = ["Pclass","Age","SibSp","Parch","Fare"] 

output = ["Survived"]



input_preproc = make_column_transformer(

    (make_pipeline(

        SimpleImputer(strategy='constant', fill_value='MISSING'),

        OneHotEncoder()),

     categorical_inputs),

    

    (make_pipeline(

        SimpleImputer(),

        StandardScaler()),

     numeric_inputs)

)
#transformátor

output_preproc = OrdinalEncoder()
#aplikácia transformátorov na dáta

X_train = input_preproc.fit_transform(df_train[categorical_inputs+numeric_inputs])

Y_train = output_preproc.fit_transform(df_train[output]).reshape(-1)



X_valid = input_preproc.transform(df_valid[categorical_inputs+numeric_inputs])

Y_valid = output_preproc.transform(df_valid[output]).reshape(-1)



X_test = input_preproc.transform(df_test[categorical_inputs+numeric_inputs])

Y_test = output_preproc.transform(df_test[output]).reshape(-1)
#transformujeme dátové typy

X_train = X_train.astype(np.float32)

Y_train = Y_train.astype(np.int64)

X_valid = X_valid.astype(np.float32)

Y_valid = Y_valid.astype(np.int64)

X_test = X_test.astype(np.float32)

Y_test = Y_test.astype(np.int64)
#vytvorenie neuronovej siete

device = "cuda" if torch.cuda.is_available() else "cpu"



num_inputs = X_train.shape[1]

num_outputs = len(np.unique(Y_train))



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(num_inputs, 70)

        self.fc2 = nn.Linear(70, 80)

        self.fc3 = nn.Linear(80, num_outputs)

        self.dropout1 = nn.Dropout(0.7)

        self.dropout2 = nn.Dropout(0.6)

        self.dropout3 = nn.Dropout(0.5)

        self.dropout4 = nn.Dropout(0.4)

        self.dropout5 = nn.Dropout(0.3)





    def forward(self, x):

        y = self.fc1(x)

        y = torch.relu(y)

        

        y = self.fc2(y)

        y = torch.relu(y)

        

        y = self.fc3(y)

        y = torch.softmax(y, dim=1)



        y = torch.relu(y)

        y = self.dropout1(y)



        y = torch.relu(y)

        y = self.dropout2(y)



        y = torch.relu(y)

        y = self.dropout3(y)

 

        y = torch.relu(y)

        y = self.dropout4(y)

        

        y = torch.relu(y)

        y = self.dropout5(y)

        

        

        return y



def predefined_array_split(X_valid, Y_valid):

    return predefined_split(

        TensorDataset(

            torch.as_tensor(X_valid),

            torch.as_tensor(Y_valid)

        )

    )

net = NeuralNetClassifier(

    Net,

    max_epochs=20,

    batch_size=-1,

    optimizer=torch.optim.Adam,

    train_split=predefined_array_split(X_valid, Y_valid),

    device=device,

    callbacks=[

        EarlyStopping(patience=10)

    ]

)
#učenie

net.fit(X_train, Y_train)
#testovanie

y_valid = net.predict(X_valid)



cm = pd.crosstab(

    output_preproc.inverse_transform(

        Y_valid.reshape(-1, 1)).reshape(-1),

    output_preproc.inverse_transform(

        y_valid.reshape(-1, 1)).reshape(-1),

    rownames=['actual'],

    colnames=['predicted']

)

print(cm)



acc = accuracy_score(Y_valid, y_valid)

print("Accuracy on valid = {}".format(acc))
#spracovanie dát na súťaž, odhodenie stlpcov

print(Xtest_df.columns.values)

Xtest_df = Xtest_df.drop('Name', axis=1,)

Xtest_df = Xtest_df.drop('Ticket', axis=1,)

Xtest_df = Xtest_df.drop('Cabin', axis=1,)

print(Xtest_df.columns.values)
#doplnenie hodnôt

Xtest_df["Age"] = Xtest_df["Age"].fillna(Xtest_df["Age"].median())

Xtest_df["Embarked"].mode()

Xtest_df["Embarked"] = Xtest_df["Embarked"].fillna("S")
#výpis

Xtest_df.head()
#transformácie

X_test1 = input_preproc.transform(Xtest_df[categorical_inputs+numeric_inputs])

Y_test1 = Xtest_df.values.reshape(-1)



X_test1 = X_test1.astype(np.float32)
#predikcia

y_test1 = net.predict(X_test1)
#odoslanie výsledkov

submission = pd.DataFrame({

        "PassengerId": Xtest_df["PassengerId"],

        "Survived": y_test1

    })

submission.to_csv('v3gender_submission.csv', index=False)