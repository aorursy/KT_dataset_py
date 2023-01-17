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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report

from sklearn.preprocessing import StandardScaler
data=pd.read_csv('/kaggle/input/equipfailstest/equip_failures_training_set.csv',na_values='na')

data.isna().sum()
test=pd.read_csv('/kaggle/input/equipfailstest/equip_failures_test_set.csv',na_values='na')

test.isna().sum()
data = data.dropna(thresh=0.95*len(data), axis=1)

test = test.dropna(thresh=0.95*len(test), axis=1)

for col in data.columns:

    if col != ('id' or 'target'):

#         data[col]=data[col].fillna(0)

        data[col]=data[col].fillna(data[col].median())



for col in test.columns:

    if col != ('id' or 'target'):

#         test[col]=test[col].fillna(0)

        test[col]=test[col].fillna(test[col].median())
collist=data.columns.tolist()

train_column=collist[2:]
X=data[train_column]

y=data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
clf = ExtraTreesClassifier(random_state=0)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)



print('ExtraTreesClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
clf = AdaBoostClassifier()

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)



print('AdaBoostClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=1)

clf.fit(X_train, y_train)



y_pred= clf.predict(X_test)

 

print('Decision Tree Classifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))



print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
clf = GradientBoostingClassifier(random_state=0)



clf.fit(X_train, y_train)



y_pred= clf.predict(X_test)



print('GradientBoostingClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
clf = RandomForestClassifier(random_state=42, max_depth=100, min_samples_leaf= 3, min_samples_split=8, n_estimators= 200)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)



print('RandomForestClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
clf = RandomForestClassifier(random_state=42, max_depth=100, min_samples_leaf= 3, min_samples_split=60, n_estimators= 500)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)



print('RandomForestClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
clf = RandomForestClassifier(random_state=0)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)

print('RandomForestClassifier accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'The accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'The F1 Score is: {f1_score(y_test,y_pred)}')
X_kaggle_test = test[train_column]

output = clf.predict(X_kaggle_test).astype(int)

df= pd.DataFrame()

df['id'] = test['id']

df['target'] = output

df.to_csv('RF_Test.csv', index=False)
# rfc=RandomForestClassifier(random_state=42)

# param_grid = { 

#     'n_estimators': [200, 500],

#     'max_features': ['auto'],

#     'max_depth' : [100,200],

#     'criterion' :['gini', 'entropy']     

# }



# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)

# CV_rfc.fit(X_train, y_train)