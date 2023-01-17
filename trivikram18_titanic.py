# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# ReadIn the training data
titanic_train = pd.read_csv("../input/train.csv")
# print (titanic_train.shape)
# print (titanic_train.columns)
print (titanic_train.info())
print (titanic_train.head())
print (titanic_train.isnull().sum())
pd.pivot_table(titanic_train, index='Pclass', values=['Age', 'Fare', 'Embarked'], 
               aggfunc={'Age': np.mean, 'Fare': np.median, 'Embarked': 'count'}, margins=True)

pd.pivot_table(titanic_train, index='Pclass', values=['Cabin', 'Fare'], aggfunc={'Cabin': 'count', 'Fare': np.median}, margins=True)

pd.pivot_table(titanic_train, index='Cabin', values=['Pclass', 'Fare', 'SibSp', 'Parch'], 
               aggfunc={'Pclass': 'count', 'Fare': np.median, 'SibSp': 'count', 'Parch': 'count'}, margins=True)

pd.pivot_table(titanic_train, index='Pclass', columns='Embarked', values='Age', aggfunc='count', margins=True)

pd.pivot_table(titanic_train, index='Cabin', columns= 'Pclass', values=['Ticket', 'Fare', 'SibSp', 'Parch'], 
               aggfunc={'Ticket':'count', 'Fare': np.mean, 'SibSp': 'count', 'Parch': 'count'})

pd.pivot_table(titanic_train, index=['Ticket', 'Cabin'], columns= 'Pclass', values=['Fare', 'SibSp', 'Parch'], 
               aggfunc={'Fare': np.mean, 'SibSp': 'count', 'Parch': 'count'})

table1 = pd.pivot_table(titanic_train, index=['Ticket', 'Cabin'], columns= 'Pclass', aggfunc={'Pclass': 'count'})
table1.head()
table1.to_csv("Titanic_table1.csv")

pd.pivot_table(titanic_train[titanic_train['Ticket']=='695'], index=['Ticket', 'Cabin'], columns= 'Pclass', aggfunc={'Pclass': 'count'})

# pd.pivot_table(titanic_train[titanic_train['Ticket'].count>1], index=['Ticket', 'Cabin'], columns= 'Pclass', aggfunc={'Pclass': 'count'})

#titanic_train[titanic_train['Ticket'].value_counts() > 1].Ticket

titanic_train['Ticket'].value_counts()
# Null Cabin
# pd.pivot_table(titanic_train[titanic_train['Cabin'].isnull()], index=['Ticket'], columns= 'Pclass', values=['Fare', 'SibSp', 'Parch'], 
#                aggfunc={'Fare': np.mean, 'SibSp': 'count', 'Parch': 'count'})

# pd.pivot_table(titanic_train[titanic_train['Cabin'].isnull()], index=['Ticket'], columns= 'Pclass', aggfunc={'Pclass': 'count'})

# titanic_train[titanic_train['Cabin'].isnull()]
# pd.crosstab(titanic_train.Cabin, titanic_train.Pclass)
# titanic_train.groupby(['Cabin']).Pclass.describe()
# ReadIn the test data
titanic_test = pd.read_csv("../input/test.csv")
print (titanic_test.shape)
print (titanic_test.columns)
# print (titanic_train.info())
# Separate out the target
y_train = titanic_train.Survived
print (type(y_train))
print (y_train.shape)
print(y_train.isnull().sum())
# Drop Survived from the dataset as it's the target. 
# Drop Name and Ticket from the dataset as they are not useful features for prediction
# X_train = titanic_train.drop(['PassengerId', 'Survived', 'Ticket'], axis=1)
X_train = titanic_train
print (X_train.shape)
print (X_train.columns)
print (X_train.info())
X_test = titanic_test
# X_test = titanic_test.drop(['PassengerId', 'Ticket'], axis=1)
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    # print (big_string)
    return np.nan

def replace_titles(x):
    title=x['salut']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
# Data Preprocessing - Training Data
# Split Name
X_train[['firstname', 'last_name']] = X_train['Name'].str.split(',',expand=True)
X_train[['salut', 'lastname', 'lastname1']] = X_train['last_name'].str.split('.',expand=True)
X_train['salut'] = X_train['salut'].str.strip()
print ("\nsalut Before:\n", X_train['salut'].value_counts())
X_train.drop(['firstname', 'last_name', 'lastname', 'lastname1'], axis=1, inplace=True)
print (X_train.columns)
X_train['salut']=X_train.apply(replace_titles, axis=1)
print ("\nsalut After:\n",X_train['salut'].value_counts())

print ("\nNull values before imputation:\n", X_train.isnull().sum())

X_train['Age'] = X_train.groupby('salut').Age.transform(lambda x: x.fillna(x.mean()))
X_train['Cabin'] = X_train['Cabin'].fillna('Null')
print (X_train['Cabin'].value_counts())
X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch']
X_train['FarePerPassenger'] = X_train['Fare']/(X_train['FamilySize'] + 1)

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Null']
X_train['Deck']=X_train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

X_train = X_train.fillna(method='ffill').fillna(method='bfill')

print ("\nNull values after imputation:\n", X_train.isnull().sum())

# X_train.to_csv("Titanic_Train_Processed.csv", index = False)

X_train.drop(['Name', 'PassengerId', 'Survived'], axis=1, inplace=True)

# One Hot Encoding - To convert categorical to binary data
X_train_dummies = pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'Cabin', 'Embarked', 'salut', 'Ticket', 'Deck'])
print ("\nShape of training dataset after One Hot Encoding\n", X_train_dummies.shape)
print (X_train_dummies.head())
# Data Preprocessing - Test Data
X_test[['firstname', 'last_name']] = X_test['Name'].str.split(',',expand=True)
X_test[['salut', 'lastname']] = X_test['last_name'].str.split('.',expand=True)
X_test['salut'] = X_test['salut'].str.strip()
print ("\nsalut Before:\n", X_test['salut'].value_counts())
X_test.drop(['Name', 'firstname', 'last_name', 'lastname'], axis=1, inplace=True)
print (X_test.columns)
X_test['salut']=X_test.apply(replace_titles, axis=1)
print ("\nsalut After:\n",X_test['salut'].value_counts())

# Imputing missing values - Test Data
print ("\nNull values before imputation:\n", X_test.isnull().sum())

X_test['Age'] = X_test.groupby('salut').Age.transform(lambda x: x.fillna(x.mean()))
X_test['Fare'] = X_test.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.median()))
X_test['Cabin'] = X_test['Cabin'].fillna('Null')
print (X_test['Cabin'].value_counts())

X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch']
X_test['FarePerPassenger'] = X_test['Fare']/(X_test['FamilySize'] + 1)
X_test['Deck']=X_test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

X_test = X_test.fillna(method='ffill').fillna(method='bfill')

print ("\nNull values after imputation:\n", X_test.isnull().sum())

# X_test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
X_test.drop(['PassengerId'], axis=1, inplace=True)

# One Hot Encoding - To convert categorical to binary data
X_test_dummies = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'Cabin', 'Embarked', 'salut', 'Ticket', 'Deck'])
print ("\nShape of test dataset after One Hot Encoding\n", X_test_dummies.shape)
print (X_test_dummies.head())
# Align the Train and Test datset for One Hot Encoding 
X_train_final, X_test_final = X_train_dummies.align(X_test_dummies, join='left', axis=1)
print (X_train_final.shape)
print (X_test_final.shape)

for col in (col for col in X_test_final.columns if X_test_final[col].isnull().any()):
    X_test_final[col] = 0

print(X_test_final.isnull().sum())
# split the data into train and evaluation data
from sklearn.model_selection import train_test_split
X, val_X, y, val_y = train_test_split(X_train_final, y_train, train_size=0.7, test_size=0.3, random_state=123, stratify=y_train)

print (X.shape)
print (val_X.shape)
print('All:', np.bincount(y_train) / float(len(y_train)) * 100.0)
print('Training:', np.bincount(y) / float(len(y)) * 100.0)
print('Test:', np.bincount(val_y) / float(len(val_y)) * 100.0)
# Using Logistic Regression for classification problem
from sklearn.linear_model import LogisticRegression
# mdlUsingLR = LogisticRegression(solver='lbfgs')
mdlUsingLR = LogisticRegression()
mdlUsingLR.fit(X, y)
print (mdlUsingLR.get_params())
print ("Count for validation data - Logistic Regression: ", np.bincount(mdlUsingLR.predict(val_X)))

print ("\nScore for training data - Logistic Regression: ", mdlUsingLR.score(X, y))
# predit and find the score for evaluation data
print ("\nScore for validation data - Logistic Regression: ", mdlUsingLR.score(val_X, val_y))
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

print ("\nf1_score for validation data - Logistic Regression: ", f1_score(val_y, mdlUsingLR.predict(val_X)))
print ("\nf1_score (average=None) for validation data - Logistic Regression: ", f1_score(val_y, mdlUsingLR.predict(val_X), average=None))
print ("\nprecision_recall_fscore for validation data - Logistic Regression: ", precision_recall_fscore_support(val_y, mdlUsingLR.predict(val_X), average=None))
print("\nAccuracy for model Logistic Regression: %.2f" % (accuracy_score(val_y, mdlUsingLR.predict(val_X)) * 100))
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, ShuffleSplit

scores = cross_val_score(mdlUsingLR, X, y)
print("\nScore for training data with default CV: ", scores)
print(np.mean(scores))

scores = cross_val_score(mdlUsingLR, X, y, cv=5)
print("\nScore for training data with CV=5: ", scores)
print(np.mean(scores))

scores = cross_val_score(mdlUsingLR, val_X, val_y, cv=5)
print("\nScore for validation data with CV=5: ", scores)
print(np.mean(scores))
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(mdlUsingLR, val_X, val_y, cv=cv)
print("\nScore for validation data with StartifiedKFold, CV=10: ", scores)
print(np.mean(scores))
# GridSearch
# Logistic Regression
from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

# mdlLRGrid = GridSearchCV(LogisticRegression(solver='newton-cg', max_iter=10000), param_grid=param_grid, cv=cv, verbose=3)
# mdlLRGrid = GridSearchCV(LogisticRegression(penalty='l1'), param_grid=param_grid, cv=cv, verbose=3)

mdlLRGrid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv, verbose=3)
mdlLRGrid.fit(X, y)
print ("\nParameters: ", mdlLRGrid.get_params)

print("\nGridSearchCV best score - Logistic Regression: ", mdlLRGrid.best_score_)
print("\nGridSearchCV best params - Logistic Regression: ", mdlLRGrid.best_params_)
print("\nGridSearchCV best estimator - Logistic Regression: ", mdlLRGrid.best_estimator_ )

# mdlLRGrid.predict(val_X)
print ("\nGridSearchCV Score for validation data - Logistic Regression: ", mdlLRGrid.score(val_X, val_y))
# Predict the y for test data
print (np.bincount(mdlUsingLR.predict(X_test_final)))
print (np.bincount(mdlLRGrid.predict(X_test_final)))
print (titanic_test.PassengerId.shape)
Survived_LR = mdlUsingLR.predict(X_test_final)
# print ("Logistic Regression:\n", Survived_LR.head())
print (Survived_LR.shape)

Survived_LRGrid = mdlLRGrid.predict(X_test_final)
# print ("Logistic Regression - Grid Search CV:\n", Survived_LRGrid.head())
print (Survived_LRGrid.shape)

Submit_LR = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': Survived_LR})
Submit_LRGrid = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': Survived_LRGrid})
Submit_LR.to_csv("Titanic_Submit_LR.csv", index = False)
Submit_LRGrid.to_csv("Titanic_Submit_LRGrid.csv", index = False)
# Model using Random Forest
from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier(n_estimators=20, min_samples_split=2, random_state=41, min_impurity_decrease=0.0, verbose=0, max_leaf_nodes=90000000)
mdlrfc = RandomForestClassifier(random_state=41)
rfc_model = mdlrfc.fit(X, y)

print ("\nParameters: ", mdlrfc.get_params)
print ("\nPrediction Score on training data: ", rfc_model.score(X, y))
print ("\nPrediction Score on validation data: ", rfc_model.score(val_X, val_y))
print("\nAccuracy for Random Forest Model: %.2f" % (accuracy_score(val_y, rfc_model.predict(val_X)) * 100))

#rfc_model.score(X_test_final, val_y)

Survived_rfc = rfc_model.predict(X_test_final)
print ("Shape of y: ", Survived_rfc.shape)

Submit_rfc = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': Survived_rfc})
Submit_rfc.to_csv("Titanic_Submit_rfc.csv", index = False)

# GridSearch
# Random Forest
param_grid = {'random_state': [41, 103, 141], 'n_estimators': [10, 20, 30], 'min_samples_split': [2, 3, 4, 5],'max_leaf_nodes':[90, 900, 9000, None] }

# rfc = RandomForestClassifier(n_estimators=20, min_samples_split=2, random_state=41, min_impurity_decrease=0.0, verbose=0, max_leaf_nodes=90000000)

RFGrid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv)
RFGrid.fit(X, y)
print ("\nParameters: ", RFGrid.get_params)

print("\nGridSearchCV best score - Random Forest: ", RFGrid.best_score_)
print("\nGridSearchCV best params - Random Forest: ", RFGrid.best_params_)
print("\nGridSearchCV best estimator - Random Forest: ", RFGrid.best_estimator_ )
# print (RFGrid.cv_results_ )

print ("\nGridSearchCV Score for validation data - Random Forest: ", RFGrid.score(val_X, val_y))
