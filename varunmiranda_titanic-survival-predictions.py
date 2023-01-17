import pandas as pd

import numpy as np

import seaborn as sns

import scipy as sp

import sklearn



from matplotlib import pyplot as plt

from scipy.stats import norm, skew

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import neural_network

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import datetime

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

train.head()
train_x = train.drop(train.columns[1],axis = 1)

train_x.head()
train_y = train[['Survived']]

train_y.head()
train_x.shape
train_x.dtypes
train_x.isna().sum()
train_x = train.drop(train.columns[[0,1,3,8,10]],axis = 1)

train_x.head()
def colsExcept(df,colsList):

  return df.loc[:, ~df.columns.isin(colsList)]  





#transofrming all object dtypes to categorical

def changeDtypes(df,from_dtype,to_dtype):

    #changes inplace, affects the passed dataFrame

#     df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).astype(to_dtype)

    df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).apply(lambda x: x.astype(to_dtype))

    

    

changeDtypes(train_x,'object','category')

train_x.dtypes
train_y.Survived.value_counts()
new_train = train_x.join(train_y) 

new_train.head()
new_train.Pclass.value_counts()
def pclass_groups(series):

    if series == 1:

        return "class 1"

    elif series == 2:

        return "class 2"

    elif series == 3:

        return "class 3"



new_train['TicketClass'] = new_train['Pclass'].apply(pclass_groups)
new_train = new_train.drop(new_train.columns[0],axis = 1)
new_train = new_train.dropna(subset=['Embarked'])

new_train["Age"].fillna(new_train["Age"].median(), inplace=True)

new_train.isna().sum()
boxplot = new_train.boxplot(column = ['Fare'])
#new_train.describe()

#new_train.Sex.value_counts()

#new_train.Age.plot.density(ind=list(range(100)))

#Age is skewed to the right, needs log transform
new_train['LogAge'] = np.log(new_train['Age'])

new_train.head()
sns.distplot(new_train['LogAge'],hist = False)
new_train = new_train.drop(['Age'],axis = 1)
sns.distplot(new_train['Fare'],hist = False)
new_train['CbrtFare'] = np.cbrt(new_train['Fare'])

new_train.head()
sns.distplot(new_train['CbrtFare'],hist = False)
new_train = new_train.drop(['Fare'],axis = 1)

new_train.head()
changeDtypes(new_train,'object','category')

new_train.dtypes
Sex_dummies = pd.get_dummies(new_train.Sex, prefix='Sex').iloc[:, 1:]

Embarked_dummies = pd.get_dummies(new_train.Embarked, prefix='Embarked').iloc[:, 1:]

TicketClass_dummies = pd.get_dummies(new_train.TicketClass, prefix='TicketClass').iloc[:, 1:]
new_train = pd.concat([new_train, Sex_dummies, Embarked_dummies, TicketClass_dummies], axis=1)

new_train = new_train.drop(['Sex','Embarked','TicketClass'],axis = 1)

new_train.head()
new_train.isna().sum()
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = new_train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
new_train_x = new_train.drop(['Survived'],axis = 1)

new_train_y = pd.DataFrame(new_train['Survived'])
X_train, X_test, Y_train, Y_test = train_test_split(new_train_x,new_train_y, test_size=0.2, random_state=42)
#Gaussian Naive Bayes - Accuracy

model = GaussianNB()

model.fit(X_train,Y_train)

Y_model = model.predict(X_test)

accuracy_score(Y_test, Y_model)
#Gaussian Naive Bayes - Confusion Matrix

mat = confusion_matrix(Y_test, Y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)

plt.xlabel('predicted value')

plt.ylabel('true value')
#Gaussian Naive Bayes - F1 Score

f1_score(Y_test, Y_model)
#Decision Tree - Accuracy

model = DecisionTreeClassifier(min_samples_split = 100)

model.fit(X_train,Y_train)

Y_model = model.predict(X_test)

accuracy_score(Y_test, Y_model)
#Decision Tree - Confusion Matrix

mat = confusion_matrix(Y_test, Y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)

plt.xlabel('predicted value')

plt.ylabel('true value')
#Decision Tree - F1 Score

f1_score(Y_test, Y_model)
#Hyper Parameter Tuning - KNN

#k_range = range(1,31)

#for k in k_range:

#    model = KNeighborsClassifier(n_neighbors=k)

#param_grid = dict(n_neighbors = k_range)

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

#grid.fit(new_train_x,new_train_y)

#print(max(grid.cv_results_['mean_test_score']))
#Hyper Parameter Tuning - Random Forest

#model = RandomForestClassifier()

#param_grid = {'n_estimators':[100,200],

#              'criterion':['gini','entropy'],

#              'max_depth':[4,5,6],

#              'max_features':['sqrt','log2',None]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

#grid.fit(new_train_x,new_train_y)

#print(max(grid.cv_results_['mean_test_score']))
#Logistic Regression - Accuracy

#model = LogisticRegression()

#model.fit(X_train,Y_train)

#Y_model = model.predict(X_test)

#accuracy_score(Y_test, Y_model)
#Logistic Regression - Confusion Matrix

#mat = confusion_matrix(Y_test, Y_model)

#sns.heatmap(mat, square=True, annot=True, cbar=False)

#plt.xlabel('predicted value')

#plt.ylabel('true value')
#Logistic Regression - F1 Score

#f1_score(Y_test, Y_model)
#Hyperparameter Tuning - Logistic Regression

#model = LogisticRegression()

#param_grid = {'penalty':['l1','l2'],

#              'C':[0.01,0.1,1,10,100]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

#grid.fit(new_train_x,new_train_y)

#print(max(grid.cv_results_['mean_test_score']))
#Hyperparameter Tuning - Support Vector Machines

#model = SVC()

#param_grid = {'C':[0.1,1,10],

#              'kernel':['linear','poly','rbf','sigmoid'],

#              'degree':[2,3,4],

#              'gamma':['auto_deprecated','scale']}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

#grid.fit(new_train_x,new_train_y)

#print(max(grid.cv_results_['mean_test_score']))
#Hyperparameter Tuning - Neural Networks (https://www.kaggle.com/hhllcks/neural-net-with-gridsearch)

#model = neural_network.MLPClassifier()

#param_grid = {'solver': ['lbfgs']}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

#grid.fit(new_train_x,new_train_y)

#print(max(grid.cv_results_['mean_test_score']))
#Hyperparameter Tuning - Adaptive Boost 

model = AdaBoostClassifier()

param_grid = {'n_estimators': [50, 100, 150, 200],

          'learning_rate': [0.5, 1.0, 1.5, 2.0]}

grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'f1')

grid.fit(new_train_x,new_train_y)

print(max(grid.cv_results_['mean_test_score']))
grid.best_params_
final_model = AdaBoostClassifier(learning_rate = 1, n_estimators = 50)

final_model.fit(new_train_x,new_train_y)
list(new_train_x)
test = pd.read_csv('../input/test.csv')

test.head()
test_x = test.drop(test.columns[[2,7,9]],axis = 1)

test_x.head()
changeDtypes(test_x,'object','category')

test_x.dtypes
new_test = test_x

new_test['TicketClass'] = new_test['Pclass'].apply(pclass_groups)

new_test.head()
new_test["Age"].fillna(train_x["Age"].median(), inplace=True)

new_test["Fare"].fillna(train_x["Fare"].median(), inplace=True)

new_test.isna().sum()
new_test['LogAge'] = np.log(new_test['Age'])

new_test.head()
new_test = new_test.drop(['Age'],axis = 1)

new_test['CbrtFare'] = np.cbrt(new_test['Fare'])

new_test = new_test.drop(['Fare'],axis = 1)

new_test.head()
new_test = new_test.drop(['Pclass'],axis = 1)

new_test = new_test.drop(['PassengerId'],axis = 1)
changeDtypes(new_test,'object','category')

new_test.dtypes
Sex_dummies = pd.get_dummies(new_test.Sex, prefix='Sex').iloc[:, 1:]

Embarked_dummies = pd.get_dummies(new_test.Embarked, prefix='Embarked').iloc[:, 1:]

TicketClass_dummies = pd.get_dummies(new_test.TicketClass, prefix='TicketClass').iloc[:, 1:]

new_test = pd.concat([new_test, Sex_dummies, Embarked_dummies, TicketClass_dummies], axis=1)

new_test = new_test.drop(['Sex','Embarked','TicketClass'],axis = 1)

new_test.head()
yhat = pd.DataFrame(final_model.predict(new_test))
yhat.shape
test['Survived'] = yhat
test = test[['PassengerId','Survived']]
test.Survived.value_counts()
#test.to_csv("Titanic_Survival_Predictions.csv",index=False)