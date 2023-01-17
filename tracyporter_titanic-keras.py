#algorithm

#https://www.kaggle.com/stefanbergstein/keras-deep-learning-on-titanic-data
#import libraries

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from itertools import combinations

from sklearn import preprocessing as pp

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import KFold

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score
# machine learning

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
# utils

import time

from datetime import timedelta



# some configuration flags and variables

verbose=0 # Use in classifier



# define random seed for reproducibility

seed = 69

np.random.seed(seed)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#read files

#Reading train file:

train = pd.read_csv('/kaggle/input/titanic/train.csv')

#Reading test file:

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train
train.info()
train.describe()
test
test.info()
test.describe()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(4)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(4)
# fill up missing values with mode

train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])

test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])



train['Fare'] = train['Fare'].fillna(train['Fare'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])



train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])



#fill missing values with median

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
train.isnull().sum().sum(), test.isnull().sum().sum()
train
test
#take title from name

train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
title=train.groupby('Title')['Survived'].sum().reset_index()

title
title1={'Capt':1, 'Col':2, 'Don':3, 'Dr':4,'Jonkheer':5, 'Lady':6, 'Major': 7, 'Master':8, 'Miss':9, 

        'Mlle':10, 'Mme':11, 'Mr':12, 'Mrs':13, 'Ms':14, 'Rev':15, 'Sir':16, 'the Countess':17, 'Dona':18}

train.Title=train.Title.map(title1)

test.Title=test.Title.map(title1)
title2=train.groupby('Title')['Survived'].sum().reset_index()

title2
train['Title'].isnull().sum().sum(), test['Title'].isnull().sum().sum()
pclass=train.groupby('Pclass')['Survived'].sum().reset_index()

pclass
pclass = train.Pclass.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

sns.barplot(x=pclass.index, y=pclass.values)

plt.show()
bins4 = [-1., 1., 2., 3. + np.inf]

names4 = ['1','2', '3']



train['Class_Range'] = pd.cut(train['Pclass'], bins4, labels=names4)

test['Class_Range'] = pd.cut(test['Pclass'], bins4, labels=names4)
class_range=train.groupby('Class_Range')['Survived'].sum().reset_index()

class_range
train['Class_Range'].isnull().sum().sum(), test['Class_Range'].isnull().sum().sum()
sex=train.groupby('Sex')['Survived'].sum().reset_index()

sex
sex1={'male':0, 'female':1}

train.Sex=train.Sex.map(sex1)

test.Sex=test.Sex.map(sex1)
sex = train.Sex.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10,4))

sns.barplot(x=sex.index, y=sex.values)

plt.show()
bins6 = [-1., 0, 1. +np.inf]

names6 = ['0','1']



train['Sex_Range'] = pd.cut(train['Sex'], bins6, labels=names6)

test['Sex_Range'] = pd.cut(test['Sex'], bins6, labels=names6)
sex1=train.groupby('Sex_Range')['Survived'].sum().reset_index()

sex1
train.Sex_Range.isnull().sum(), test.Sex_Range.isnull().sum()
age=train.groupby('Age')['Survived'].sum().reset_index()

age
plt.figure(figsize=(10,6))

plt.title("Ages Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Age"])

plt.show()
age18_25 = train.Age[(train.Age <= 25) & (train.Age >= 18)]

age26_35 = train.Age[(train.Age <= 35) & (train.Age >= 26)]

age36_45 = train.Age[(train.Age <= 45) & (train.Age >= 36)]

age46_55 = train.Age[(train.Age <= 55) & (train.Age >= 46)]

age55above = train.Age[train.Age >= 56]



x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]



plt.figure(figsize=(15,6))

sns.barplot(x=x, y=y, palette="rocket")

plt.title("Number of Survivors and Ages")

plt.xlabel("Age")

plt.ylabel("Number of Survivors")

plt.show()
bins = [0., 18., 35., 64., 65.+ np.inf]

names = ['child','young adult', 'middle aged', 'pensioner']



train['Age_Range'] = pd.cut(train['Age'], bins, labels=names)

test['Age_Range'] = pd.cut(test['Age'], bins, labels=names)
age_range=train.groupby('Age_Range')['Survived'].sum().reset_index()

age_range
age_range1={'child':1,'young adult':2, 'middle aged':3, 'pensioner': 4}

train.Age_Range=train.Age_Range.map(age_range1)

test.Age_Range=test.Age_Range.map(age_range1)
train.Age_Range.isnull().sum(), test.Age_Range.isnull().sum()
age_range=train.groupby('Age_Range')['Survived'].sum().reset_index()

age_range
family=train.groupby('SibSp')['Survived'].sum().reset_index()

family
plt.figure(figsize=(10,6))

plt.title("Family Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["SibSp"])

plt.show()
bins2 = [-1., 0., 1., 2., 3., 4., 5., 8.+ np.inf]

names2 = ['0','1', '2', '3', '4', '5', '8']



train['Family_Range'] = pd.cut(train['SibSp'], bins2, labels=names2)

test['Family_Range'] = pd.cut(test['SibSp'], bins2, labels=names2)
family1=train.groupby('Family_Range')['Survived'].sum().reset_index()

family1
parch=train.groupby('Parch')['Survived'].sum().reset_index()

parch
plt.figure(figsize=(10,6))

plt.title("Parch Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Parch"])

plt.show()
bins3 = [-1., 0., 1., 2., 3., 4., 5., 6.+ np.inf]

names3 = ['0','1', '2', '3', '4', '5', '6']



train['Parch_Range'] = pd.cut(train['SibSp'], bins3, labels=names3)

test['Parch_Range'] = pd.cut(test['SibSp'], bins3, labels=names3)
parch1=train.groupby('Parch_Range')['Survived'].sum().reset_index()

parch1
fare=train.groupby('Fare')['Survived'].sum().reset_index()

fare
plt.figure(figsize=(10,6))

plt.title("Fare Frequency")

sns.axes_style("dark")

sns.violinplot(y=train["Fare"])

plt.show()
bins0 = [-1., 100., 200., 300., 400., 500.+ np.inf]

names0 = ['0-99', '100-199', '200-299', '300-399', '400+']



train['Fare_Range'] = pd.cut(train['Fare'], bins0, labels=names0)

test['Fare_Range'] = pd.cut(test['Fare'], bins0, labels=names0)
fare_range=train.groupby('Fare_Range')['Survived'].sum().reset_index()

fare_range
fare_range1={'0-99':1, '100-199':2, '200-299': 3, '300-399':0, '400+':5}

train.Fare_Range=train.Fare_Range.map(fare_range1)

test.Fare_Range=test.Fare_Range.map(fare_range1)
train.Fare_Range.isnull().sum(), test.Fare_Range.isnull().sum()
embark=train.groupby('Embarked')['Survived'].sum().reset_index()

embark
embark1={'C':1, 'Q':2, 'S':3}

train.Embarked=train.Embarked.map(embark1)

test.Embarked=test.Embarked.map(embark1)
bins1 = [-1., 1., 2., 3.+ np.inf]

names1 = ['C', 'Q', 'S']



train['Embarked_Range'] = pd.cut(train['Embarked'], bins1, labels=names1)

test['Embarked_Range'] = pd.cut(test['Embarked'], bins1, labels=names1)
train.Embarked_Range.isnull().sum(), test.Embarked_Range.isnull().sum()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(3)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(3)
train.isnull().sum().sum(), test.isnull().sum().sum()
train
test
train.dtypes
test.dtypes
#create a heatmap to correlate survival

plt.figure(figsize=(6,4))

cmap=train.corr()

sns.heatmap(cmap, annot=True)

# Feature selection: remove variables no longer containing relevant information

drop_elements = ['Name', 'Ticket', 'Cabin', 'Title']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)
y = train["Survived"]

features = ["Class_Range", "Sex_Range", "Embarked_Range", "Age_Range", "Family_Range", "Parch_Range", "Fare_Range"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
X
X_test
#split train set for training and testing

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, stratify=y, test_size=0.25, random_state=101)
# distribution in training set

Y_train.value_counts(normalize=True)
# distribution in validation set

Y_validation.value_counts(normalize=True)
#shape of training set

X_train.shape, Y_train.shape
#shape of validation set

X_validation.shape, Y_validation.shape
#Simple network using keras

def create_model(optimizer='adam', init='uniform'):

    # create model

    if verbose: print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )

    model = Sequential()

    model.add(Dense(16, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))

    model.add(Dense(8, kernel_initializer=init, activation='relu'))

    model.add(Dense(4, kernel_initializer=init, activation='relu'))

    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
#grid search

run_gridsearch = False



if run_gridsearch:

    

    start_time = time.time()

    if verbose: print (time.strftime( "%H:%M:%S " + "GridSearch started ... " ) )

    optimizers = ['rmsprop', 'adam']

    inits = ['glorot_uniform', 'normal', 'uniform']

    epochs = [50, 100, 200, 400]

    batches = [5, 10, 20]

    

    model = KerasClassifier(build_fn=create_model, verbose=verbose)

    

    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)

    grid_result = grid.fit(X, Y)

    

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    if verbose: 

        for mean, stdev, param in zip(means, stds, params):

            print("%f (%f) with: %r" % (mean, stdev, param))

        elapsed_time = time.time() - start_time  

        print ("Time elapsed: ",timedelta(seconds=elapsed_time))

        

    best_epochs = grid_result.best_params_['epochs']

    best_batch_size = grid_result.best_params_['batch_size']

    best_init = grid_result.best_params_['init']

    best_optimizer = grid_result.best_params_['optimizer']

    

else:

    # pre-selected paramters

    best_epochs = 200

    best_batch_size = 5

    best_init = 'glorot_uniform'

    best_optimizer = 'rmsprop'
# Create a classifier with best parameters

model_pred = KerasClassifier(build_fn=create_model, optimizer=best_optimizer, init=best_init, epochs=best_epochs, 

                             batch_size=best_batch_size, verbose=verbose)

model_pred.fit(X, y)



# Predict 'Survived'

prediction = model_pred.predict(X_test)
submission = pd.DataFrame({

    'PassengerId': test.PassengerId,

    'Survived': prediction[:,0],

})



submission.sort_values('PassengerId', inplace=True)    

submission.to_csv('submission-simple-cleansing.csv', index=False)

submission