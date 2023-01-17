# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data['Sex'] == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#Looking at what columns there are null entries

train_data.isna().sum()
#What are the dtypes of each column?

train_data.info()
#Defining a new feature based on whether or not 'cabin' is missing

train_data['Cabin_is_known'] = train_data['Cabin'].notnull().astype('int')

#Dropping 'PassengerId','Name','Ticket', and 'Cabin' columns

train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)



train_data.head()
#Inspecting correlations between features

train_data.corr()
#Filling the missing values in the 'Age' column

train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

#Dropping the remaining rows with missing values (in the 'Embarked' column)

train_data = train_data.dropna()



train_data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



#Distribution of age between different genres

g = sns.FacetGrid(train_data, col="Sex")

g = g.map(plt.hist, "Age")
sns.catplot(x='Sex',y='Survived',data=train_data,kind='bar',hue='Cabin_is_known')

plt.show()
#Getting dummy variables from the categorical variables

X_train = pd.get_dummies(1*train_data,drop_first=True)

X_train.head()
#Creating numpy arrays for the training examples (X) and corresponding labels (y)

y,X = X_train.values[:,0],X_train.values[:,1:]
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV



#Initialazing the pipeline with the steps corresponding to normalizing the features and then applying the model

steps = [('scaler', StandardScaler()), ('logreg',LogisticRegression(solver='liblinear'))]

pipeline = Pipeline(steps)



#Defining the set of hyperparameters to be tested. We shall select the best combination

C = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]

max_iter=[50,100,150,200,250,300,350,400,450,500]

tol=[0.00001,0.00003,0.0001,0.0003,0.001]

penalty = ['l1','l2']

param_grid = dict(logreg__C=C,logreg__max_iter=max_iter,logreg__tol=tol,logreg__penalty = penalty)



#Defining and fitting the model to the data.

grid_model = GridSearchCV(pipeline, param_grid=param_grid, cv=5)



grid_model_result = grid_model.fit(X, y)
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))



y_pred = grid_model_result.predict(X)

# Print the confusion matrix of the logreg model

print(confusion_matrix(y,y_pred))
test_set = test_data.copy()

test_set['Cabin_is_known'] = test_set['Cabin'].notnull().astype('int')

test_set = test_set.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test_set['Age'] = test_set.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

test_set = test_set.fillna(test_set.mean())

test_set = pd.get_dummies(test_set,drop_first=True)

test_set.head()
#Predicting the labels of the test set and saving the submission file

predictions = grid_model_result.predict(test_set.values).astype(int)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('logreg_submission.csv', index=False)

print("Your submission was successfully saved!")
#Taking a look at the submission file.

pd.read_csv('logreg_submission.csv').head()
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV



#Initialazing the pipeline with the steps corresponding to normalizing the features and then applying the model

steps = [('scaler', StandardScaler()), ('svm',SVC())]

pipeline = Pipeline(steps)



#Defining the set of hyperparameters to be tested. We shall select the best combination

C = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]

gamma = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]

param_grid = dict(svm__C=C,svm__gamma=gamma)



#Defining and fitting the model to the data.

grid_model = GridSearchCV(pipeline, param_grid=param_grid, cv=5)



grid_model_result = grid_model.fit(X, y)
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))



y_pred = grid_model_result.predict(X)

# Print the confusion matrix of the SVM model

print(confusion_matrix(y,y_pred))
#Predicting the labels of the test set and saving the submission file

predictions = grid_model_result.predict(test_set.values).astype(int)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('svm_submission.csv', index=False)

print("Your submission was successfully saved!")
#Taking a look at the submission file.

pd.read_csv('svm_submission.csv').head()
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV



#Initialazing the pipeline with the steps corresponding to normalizing the features and then applying the model

steps = [('scaler', StandardScaler()), ('rf',RandomForestClassifier())]

pipeline = Pipeline(steps)



#Defining the set of hyperparameters to be tested. We shall select the best combination

max_depth = [3,4,5]

n_estimators = [100,200]

min_samples_split = [2,4,5]

min_samples_leaf = [1,3,5]

param_grid = dict(rf__max_depth=max_depth,rf__n_estimators=n_estimators,\

                  rf__min_samples_split=min_samples_split,\

                 rf__min_samples_leaf=min_samples_leaf)



#Defining and fitting the model to the data.

grid_model = GridSearchCV(pipeline, param_grid=param_grid, cv=5)



grid_model_result = grid_model.fit(X, y)
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))



y_pred = grid_model_result.predict(X)

# Print the confusion matrix of the SVM model

print(confusion_matrix(y,y_pred))
#Predicting the labels of the test set and saving the submission file

predictions = grid_model_result.predict(test_set.values).astype(int)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('rf_submission.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import RandomizedSearchCV



#Initialazing the pipeline with the steps corresponding to normalizing the features and then applying the model

steps = [('scaler', StandardScaler()), ('xgboost',xgb.XGBClassifier())]

pipeline = Pipeline(steps)



#Defining the set of hyperparameters to be tested. We shall select the best combination

learning_rate = [0.05,0.15,0.3]

max_depth = [3,4,5]

n_estimators = [50,100,200]

subsample = [0.8,0.9]

colsample_bytree = [0.8,0.9,1]

param_grid = dict(xgboost__learning_rate=learning_rate,xgboost__max_depth=max_depth\

                  ,xgboost__n_estimators=n_estimators,xgboost__subsample=subsample,\

                 xgboost__colsample_bytree=colsample_bytree)



#Defining and fitting the model to the data.

grid_model = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=5,n_iter=100)



grid_model_result = grid_model.fit(X, y)
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))



y_pred = grid_model_result.predict(X)

# Print the confusion matrix of the SVM model

print(confusion_matrix(y,y_pred))
#Predicting the labels of the test set and saving the submission file

predictions = grid_model_result.predict(test_set.values).astype(int)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('xgboost_submission.csv', index=False)

print("Your submission was successfully saved!")
#Taking a look at the submission file.

pd.read_csv('xgboost_submission.csv').head()