# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

data.head()



#Submission Data

test_data.head()
fig = plt.figure(figsize = (30,20))

ax = fig.gca()

hist = data.hist(ax=ax)
data.describe()
data['Cabin'].head() #Too much missing data to augment with the mean
#replace male female with 1 and 0

data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}) 

#fill in missing data in age

data['Age'].fillna(data['Age'].mean(), inplace=True)

#fill in missing data in embarked

data['Embarked'].fillna( method ='ffill', inplace = True) 

#one hot encode embarked

one_hot_columns = pd.get_dummies(data['Embarked'],prefix=None)

# use pd.concat to join the new columns with your original dataframe

data = pd.concat([data,one_hot_columns],axis=1)

# now drop the original 'country' column (you don't need it anymore)

data.drop(['Embarked'],axis=1, inplace=True)



# Convert ticket to integer

import re



def ticket_to_float(ticket_str):

    ticket_numbers_only = ''.join(i for i in ticket_str if i.isdigit())

    if ticket_numbers_only is '':

        return 0

    return int(ticket_numbers_only)



data['Ticket'] = data['Ticket'].apply(ticket_to_float)



#Submission Data



#replace male female with 1 and 0

test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0}) 

#fill in missing data in age

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

#fill in missing data in fare

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

#fill in missing data in embarked

test_data['Embarked'].fillna( method ='ffill', inplace = True) 

#one hot encode embarked

one_hot_columns = pd.get_dummies(test_data['Embarked'],prefix=None)

# use pd.concat to join the new columns with your original dataframe

test_data = pd.concat([test_data,one_hot_columns],axis=1)

# now drop the original 'country' column (you don't need it anymore)

test_data.drop(['Embarked'],axis=1, inplace=True)

# Convert ticket to integer

test_data['Ticket'] = test_data['Ticket'].apply(ticket_to_float)



data.head()
if data['Survived'].isnull().values.any():

    print("Missing Values Survived")

if data['Pclass'].isnull().values.any():

    print("Missing Values Pclass")

if data['Sex'].isnull().values.any():

    print("Missing Values Sex")

if data['Age'].isnull().values.any():

    print("Missing Values Age")

if data['SibSp'].isnull().values.any():

    print("Missing Values SibSp")

if data['Parch'].isnull().values.any():

    print("Missing Values Parch")

if data['Fare'].isnull().values.any():

    print("Missing Values Fare")

    

#Submission Data

if test_data['Pclass'].isnull().values.any():

    print("Missing Values Pclass")

if test_data['Sex'].isnull().values.any():

    print("Missing Values Sex")

if test_data['Age'].isnull().values.any():

    print("Missing Values Age")

if test_data['SibSp'].isnull().values.any():

    print("Missing Values SibSp")

if test_data['Parch'].isnull().values.any():

    print("Missing Values Parch")

if test_data['Fare'].isnull().values.any():

    print("Missing Values Fare")
feature_columns = [

    'Pclass',

    'Sex',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'C',

    'Q',

    'S'

]



data[feature_columns].head()
from sklearn import preprocessing



#Training Data

x = data[feature_columns].values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

data[feature_columns] = pd.DataFrame(x_scaled)



#Submission Data

x_test_data = test_data[feature_columns].values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled_test_data = min_max_scaler.fit_transform(x_test_data)

test_data[feature_columns] = pd.DataFrame(x_scaled_test_data)



fig = plt.figure(figsize = (30,20))

ax = fig.gca()

hist = test_data[feature_columns].hist(ax=ax)
f = plt.figure(figsize=(19, 15))

plt.matshow(data[feature_columns].corr(), fignum=f.number)

plt.xticks(range(data[feature_columns].shape[1]), data[feature_columns].columns, fontsize=14, rotation=45)

plt.yticks(range(data[feature_columns].shape[1]), data[feature_columns].columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
from sklearn.model_selection import train_test_split



X = data[feature_columns]

y = data['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=0)



X_train.shape

from sklearn.linear_model import LogisticRegression 

from sklearn.neural_network import MLPClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Logistic Regression

log_reg = LogisticRegression(solver='sag', random_state=0)

log_reg.fit(X_train, y_train)



# Neural Net

n_net = MLPClassifier(hidden_layer_sizes=(4,4,4),max_iter=500)

n_net.fit(X_train, y_train)



# Support Vector Machine

svmC = svm.SVC(kernel='linear')

svmC.fit(X_train, y_train)



# kNN

k_NN = KNeighborsClassifier(n_neighbors=3)

k_NN.fit(X_train, y_train)



# Random Forest

rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



if y_test is not None:



    y_pred_log_reg = log_reg.predict(X_test)



    confusion_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)



    sn.heatmap(confusion_matrix_log_reg, annot=True, cmap='Blues', fmt='g')



    target_names = ['Survived', 'Not Survived']



    print(classification_report(y_test, y_pred_log_reg, target_names=target_names))



if y_test is not None:

    y_pred_n_net = n_net.predict(X_test)



    confusion_matrix_n_net = confusion_matrix(y_test, y_pred_n_net)



    sn.heatmap(confusion_matrix_n_net, annot=True, cmap='Blues', fmt='g')



    target_names = ['Survived', 'Not Survived']



    print(classification_report(y_test, y_pred_n_net, target_names=target_names))



    print("Accuracy:",accuracy_score(y_test, y_pred_log_reg))
if y_test is not None:

    y_pred_svm = svmC.predict(X_test)



    confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)



    sn.heatmap(confusion_matrix_svm, annot=True, cmap='Blues', fmt='g')



    target_names = ['Survived', 'Not Survived']



    print(classification_report(y_test, y_pred_svm, target_names=target_names))
if y_test is not None:

    y_pred_k_NN = k_NN.predict(X_test)



    confusion_matrix_k_NN = confusion_matrix(y_test, y_pred_k_NN)



    sn.heatmap(confusion_matrix_k_NN, annot=True, cmap='Blues', fmt='g')



    target_names = ['Survived', 'Not Survived']



    print(classification_report(y_test, y_pred_k_NN, target_names=target_names))
if y_test is not None:

    y_pred_rfc = rfc.predict(X_test)



    confusion_matrix_rfc = confusion_matrix(y_test, y_pred_rfc)



    sn.heatmap(confusion_matrix_rfc, annot=True, cmap='Blues', fmt='g')



    target_names = ['Survived', 'Not Survived']



    print(classification_report(y_test, y_pred_rfc, target_names=target_names))
if y_test is not None:

    print("Accuracy Logistic Regression:",accuracy_score(y_test, y_pred_log_reg))

    print("Accuracy Neural Net:",accuracy_score(y_test, y_pred_n_net))

    print("Accuracy Support Vector Machine:",accuracy_score(y_test, y_pred_svm))

    print("Accuracy kNN:",accuracy_score(y_test, y_pred_k_NN))

    print("Accuracy Random Forest:",accuracy_score(y_test, y_pred_rfc))
X_train=data[feature_columns + ["PassengerId"]]

X_test=test_data[feature_columns  + ["PassengerId"]]

y_train=data['Survived']

y_test=None



id_nr = X_test["PassengerId"]

X_train = X_train.drop(columns=["PassengerId"])

X_test = X_test.drop(columns=["PassengerId"])
import csv



# Random Forest

rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)



sb = rfc.predict(X_test)



submission = pd.DataFrame({

    'PassengerId':id_nr,

    'Survived':sb

})



submission.head()
submission.to_csv('csv_to_submit.csv', index = False)