# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import datasets, linear_model

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

%matplotlib inline

import matplotlib.pyplot as plt



dfgender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

dftest = pd.read_csv("/kaggle/input/titanic/test.csv")

dftrain = pd.read_csv("/kaggle/input/titanic/train.csv")
dftrain.info()
dfy=dftrain['Survived']

dfy.shape
dfx=dftrain.drop('Survived', axis=1)

dfx=dfx.drop('Name', axis=1)

dfxtest=dftest.drop('Name', axis=1)

print(dfx.shape)

print(dfxtest.shape)


one_hot_encoded_dfx = pd.get_dummies(dfx)

one_hot_encoded_dfxtest = pd.get_dummies(dfxtest)

final_train, final_test = one_hot_encoded_dfx.align(

                                    one_hot_encoded_dfxtest,

                                    join='left', 

                                    axis=1)

print(one_hot_encoded_dfx.shape)

print(one_hot_encoded_dfxtest.shape)

print(final_train.shape)

print(final_test.shape)
imputed_X_train_plus = final_train.copy()

imputed_X_test_plus = final_test.copy()



cols_with_missing = (col for col in final_train.columns 

                                 if final_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)



print(imputed_X_train_plus.shape)

print(imputed_X_test_plus.shape)

X_train, X_test, y_train, y_test = train_test_split(imputed_X_train_plus, dfy, test_size=0.33, random_state=42)

tree = DecisionTreeClassifier()

grad = GradientBoostingRegressor()

bag =  BaggingRegressor()

logistic_reg = LogisticRegression()

rndforestes = RandomForestClassifier()

mlp = MLPClassifier()

svc = LinearSVC()

gauss = GaussianNB()
tree.fit(X_train,y_train)

grad.fit(X_train,y_train)

bag.fit(X_train,y_train)

logistic_reg.fit(X_train,y_train)

rndforestes.fit(X_train,y_train)

mlp.fit(X_train,y_train)

svc.fit(X_train,y_train)

gauss.fit(X_train,y_train)
print(tree.score(X_test,y_test)*100)

print(grad.score(X_test,y_test)*100)

print(bag.score(X_test,y_test)*100)

print(logistic_reg.score(X_test,y_test)*100)

print(rndforestes.score(X_test,y_test)*100)

print(mlp.score(X_test,y_test)*100)

print(svc.score(X_test,y_test)*100)

print(gauss.score(X_test,y_test)*100)
models = pd.DataFrame({

    'Model': ['DecisionTreeClassifier', 'GradientBoostingRegressor', 'BaggingRegressor' , 'LogisticRegression' ,'RandomForestClassifier', 'MLPClassifier', 

              'LinearSVC', 'GaussianNB'],

    'Score': [tree.score(X_test,y_test)*100, grad.score(X_test,y_test)*100, bag.score(X_test,y_test)*100, logistic_reg.score(X_test,y_test)*100, rndforestes.score(X_test,y_test)*100,

              mlp.score(X_test,y_test)*100, svc.score(X_test,y_test)*100, gauss.score(X_test,y_test)*100]})

models.sort_values(by='Score', ascending=False)
result = logistic_reg.predict(imputed_X_test_plus)

result.shape
dfresult = dfgender

dfresult["Survived"] = result

dfresult.tail()
dfgender.tail()