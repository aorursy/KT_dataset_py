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
gs = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/gender_submission.csv')

dftrain = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')

dftest = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/test.csv')



df1 = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')
gs.head()
dftrain.head()
dftest.head()
import seaborn as sns

import matplotlib.pyplot as plt
for i in dftrain:

    print(i, dftrain[i].isna().sum())
dftrain = dftrain.drop(columns = ['Cabin', 'PassengerId'])
dftrain['Age'].fillna(int(dftrain['Age'].mean()), inplace = True)



plt.hist(dftrain['Age'])
dftrain.dropna(inplace = True)
dftrain.head()
def plotBarChart(data,col,label):

    g = sns.FacetGrid(data, col=col)

    g.map(plt.hist, label, bins=10)



for val in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:

    plotBarChart(dftrain,'Survived',val)
dftrain[dftrain['Fare'] == 0]['Age']
names = list(dftrain['Name'])



titles = list(map(lambda x: x.split(',')[1].split('.')[0], names))



print (len(dict.fromkeys(titles)))



for i in dict.fromkeys(titles):

    print(i, titles.count(i))

    

    

dftrain['Titles'] = titles



dftrain = pd.get_dummies(dftrain.drop(columns = ['Name', 'Ticket']))



dftrain.columns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.linear_model import LogisticRegression as lr

from sklearn.model_selection import train_test_split as tts



x = dftrain.drop(columns = ['Survived'])

y = dftrain['Survived']



x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)



regressor = dtc()



regressor1 = lr()



regressor.fit(x_train, y_train)



print(regressor.score(x_test, y_test))



regressor1.fit(x_train, y_train)

print(regressor1.score(x_test, y_test))
cols = ['Titles_ Capt','Titles_ Col', 'Titles_ Don', 'Titles_ Dr', 'Titles_ Jonkheer',

       'Titles_ Lady', 'Titles_ Major', 'Titles_ Master', 'Titles_ Miss',

       'Titles_ Mlle', 'Titles_ Mme', 'Titles_ Mr', 'Titles_ Mrs',

       'Titles_ Ms', 'Titles_ Rev', 'Titles_ Sir', 'Titles_ the Countess']
from sklearn.ensemble import RandomForestClassifier

x = dftrain.drop(columns = ['Survived', 'Sex_female', 'Fare', 'Age'] + cols)

y = dftrain['Survived']



print(x.columns)



x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)



regressor = RandomForestClassifier()



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test), regressor.score(x_train, y_train)
from sklearn.model_selection import GridSearchCV



grid_values = {'n_estimators': [100],'min_samples_split':[4],'min_samples_leaf':[2]}



grid_rfc = GridSearchCV(estimator=regressor, param_grid = grid_values, cv= 3,n_jobs=-1)



grid_rfc.fit(x_train, y_train)



grid_rfc.score(x_test, y_test), grid_rfc.score(x_train, y_train)
from sklearn.ensemble import AdaBoostClassifier as abc

import time



x = dftrain.drop(columns = ['Survived', 'Sex_female', 'Fare'])

y = dftrain['Survived']



test_accuracy = []

train_accuracy = []



t = time.time()



for i in range(1, 51):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)



    regressor = abc()



    regressor.fit(x_train, y_train)



    test_accuracy.append(regressor.score(x_test, y_test))

    train_accuracy.append(regressor.score(x_train, y_train))

    

print ('Test:', sum(test_accuracy)/len(test_accuracy), 'Train:', sum(train_accuracy)/len(train_accuracy))

print (time.time() - t)
from xgboost import XGBClassifier as xgbc



t = time.time()



for i in range(1, 51):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)



    regressor = xgbc()



    regressor.fit(x_train, y_train)



    test_accuracy.append(regressor.score(x_test, y_test))

    train_accuracy.append(regressor.score(x_train, y_train))

    

print ('Test:', sum(test_accuracy)/len(test_accuracy), 'Train:', sum(train_accuracy)/len(train_accuracy))

print (time.time() - t)
from sklearn.ensemble import GradientBoostingClassifier as gbc



t = time.time()



for i in range(1, 51):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)



    regressor = gbc()



    regressor.fit(x_train, y_train)



    test_accuracy.append(regressor.score(x_test, y_test))

    train_accuracy.append(regressor.score(x_train, y_train))

    

print ('Test:', sum(test_accuracy)/len(test_accuracy), 'Train:', sum(train_accuracy)/len(train_accuracy))

print (time.time() - t)
from sklearn import metrics



y_pred_proba = regressor.predict_proba(x_test)[::,1]

y_pred_proba1 = regressor.predict_proba(x_train)[::,1]



fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

fpr1, tpr1, _ = metrics.roc_curve(y_train,  y_pred_proba1)



auc = metrics.roc_auc_score(y_test, y_pred_proba)

auc1 = metrics.roc_auc_score(y_train, y_pred_proba1)



plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot(fpr1,tpr1,label="data 2, auc="+str(auc1))



plt.legend(loc=4)



plt.show()
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

actual = y_test

predicted = regressor.predict(x_test)

results = confusion_matrix(actual, predicted) 

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(actual, predicted))

print ('Report : ')

print (classification_report(actual, predicted))