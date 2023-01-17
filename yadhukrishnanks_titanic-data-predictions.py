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

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,confusion_matrix

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint as sp_randint
df1=pd.read_csv('../input/titanic/train.csv')

df2=pd.read_csv('../input/titanic/gender_submission.csv')

df3=pd.read_csv('../input/titanic/test.csv')
plt.figure(figsize=(10,5))

sns.heatmap(df1.corr(),annot=True)

plt.show()
df1.isnull().sum()/len(df1)
df1.info()
df1=df1.drop(columns=['Name'])

#dropping cabin volumn as 77% is null values

df1=df1.drop(columns=['Cabin'])

df1=df1.drop(columns=['Ticket'])



df3=df3.drop(columns=['Name'])

#dropping cabin volumn as 77% is null values

df3=df3.drop(columns=['Cabin'])

df3=df3.drop(columns=['Ticket'])
df2
df1.fillna(df1.median(),inplace=True)

df2

df3.fillna(df1.median(),inplace=True)
df1=pd.get_dummies(df1)

df3=pd.get_dummies(df3)

df1.isnull().sum()
x_train=df1.drop(columns=['Survived'],axis=1)

y_train=df1.Survived

x_test=df3

y_test=df2.drop(columns=['PassengerId'])
model=DecisionTreeClassifier()

model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)

y_pred_test = model.predict(x_test)



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred_test, y_test))
df2['pred']=y_pred_test
confusion_matrix(y_test, y_pred_test)
## tuning model

model = DecisionTreeClassifier(random_state = 1)



params = {'max_depth' : [2,3,4,5,6,7,8],

        'min_samples_split': [2,3,4,5,6,7,8,9,10],

        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}



gsearch = GridSearchCV(model, param_grid = params, cv = 3)



gsearch.fit(x_train,y_train)



print(gsearch.best_params_)
model=DecisionTreeClassifier(**gsearch.best_params_)

model.fit(x_train, y_train)



y_pred_train = model.predict(x_train)

y_pred_test = model.predict(x_test)



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred_test, y_test))
confusion_matrix(y_test, y_pred_test)
## Random forest classifier



rfc = RandomForestClassifier()



params = {'n_estimators': sp_randint(5,25),

    'criterion': ['gini', 'entropy'],

    'max_depth': sp_randint(2, 10),

    'min_samples_split': sp_randint(2,20),

    'min_samples_leaf': sp_randint(1, 20),

    'max_features': sp_randint(2,11)}



rand_search_rfc = RandomizedSearchCV(rfc, param_distributions=params,

                                 cv=3, random_state=1)



rand_search_rfc.fit(x_train,y_train)

print(rand_search_rfc.best_params_)
rfc = RandomForestClassifier(**rand_search_rfc.best_params_)



rfc.fit(x_train, y_train)

y_pred_train_r = rfc.predict(x_train)

y_pred_test_r = rfc.predict(x_test)



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train_r, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred_test_r, y_test))
confusion_matrix(y_test, y_pred_test)
##Stacked algorithm



lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(**rand_search_rfc.best_params_)

model=DecisionTreeClassifier(**gsearch.best_params_)



clf = VotingClassifier(estimators=[('lr',lr), ('rfc',rfc), ('dt',model)], 

                       voting='soft')

clf.fit(x_train, y_train)



y_pred_train = clf.predict(x_train)

y_pred_test = clf.predict(x_test)



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred_test, y_test))
confusion_matrix(y_test, y_pred_test)
Submission = pd.DataFrame({ 'PassengerId': df2.PassengerId,

                            'Survived': y_pred_test_r })

Submission.to_csv("StackingSubmission.csv", index=False)
Submission
df2