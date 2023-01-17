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
import matplotlib.pyplot as plt #for plotting
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
def read_data(path):
    '''read data from path and return a pandas dataframe'''
    df = pd.read_csv(path)
    return df

train_path = '../input/train.csv'
test_path = '../input/test.csv'

train = read_data(train_path)
test = read_data(test_path)
train.head()
train.describe().T
test.describe().T
train.info()
test.info()
train.columns
def map_sex(df):
    '''map female to 1 and male to 0 and return the dataframe'''
    df['Sex'] = df['Sex'].map({'female': 1, 'male' : 0}).astype(int)
    return df
train = map_sex(train)
test = map_sex(test)
train.head()
print('The most occurence values in train : {}'.format(train['Embarked'].mode()))
print('The most occurrence values in test : {}'.format(test['Embarked'].mode()))
print('Train : {}'.format(train['Embarked'].unique()))
print('Test : {}'.format(test['Embarked'].unique()))
mos_freq = train['Embarked'].mode()
def fill_missing_emberked(df):
    '''fill the values values in embarked column with the mode'''
    df['Embarked'] = df['Embarked'].fillna(mos_freq)
    return df

train = fill_missing_emberked(train)
test = fill_missing_emberked(test)
    
#map embarked to numeric representation
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
train.describe()
test.describe()
train['Age'].hist(bins=50) #check the Age duistribution
round(train['Age'].mean())
print(train.shape)
print(test.shape)
from pandas.plotting import scatter_matrix
scatter_matrix(train, alpha=0.2, figsize=(6, 6), diagonal='kde')

train['Age'].fillna(round(train['Age'].mean()), inplace=True)
test['Age'].fillna(round(test['Age'].mean()), inplace=True)
train.head()
train.describe()
test.describe()
train['Embarked'].isnull().sum()

train['Embarked'].fillna(0, inplace=True)
test['Embarked'].fillna(0, inplace=True)
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
train_df = train[feature_columns]
train_df.head()
scatter_matrix(train_df,alpha=0.2,diagonal='kde')
test_df = test[feature_columns]
test_df.head()
X_train = train[feature_columns]
y_train = train['Survived']
X_test = test[feature_columns]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
X_train1,X_val, y_train1,y_val = train_test_split(X_train,y_train, random_state=42)
print(X_train1.shape, y_train1.shape)
print(X_val.shape, y_val.shape)
#baseline model to check the oob_score
rdf = RandomForestClassifier(oob_score=True, random_state=42,n_estimators=20,max_features = 2,
                             n_jobs =-1,max_depth=5)
#we use all the test data set to check the oob score
rdf.fit(X_train,y_train)
print(rdf.oob_score_)
cross_val = cross_val_score(rdf,X_train,y_train, scoring='accuracy')
cross_val.mean()
from sklearn.metrics import accuracy_score

depth_option = [1,2,3,4,5,6,7,8,9,10]
train_score =[]
val_score = []
for depth in depth_option:
    model = RandomForestClassifier(n_estimators=100,max_depth=depth)
    model.fit(X_train1,y_train1)
    y_train_pred = model.predict(X_train1)
    y_val_pred = model.predict(X_val)
    train_score.append(accuracy_score(y_train1,y_train_pred))
    val_score.append(accuracy_score(y_val,y_val_pred))
    print('max_depth : {}'.format(depth))
    print('train score : {}'.format(accuracy_score(y_train1,y_train_pred)))
    print('test score : {}'.format(accuracy_score(y_val,y_val_pred)))
    
plt.plot(depth_option,train_score, label='train score')
plt.plot(depth_option,val_score, label='val score')
plt.xlabel('max_depth option')
plt.ylabel('accuracy score')
plt.legend()
plt.show()



max_features_option = [1,2,3,4,5,6,7]
train_score =[]
val_score = []
for features in max_features_option:
    model = RandomForestClassifier(n_estimators=100,max_depth=3, max_features=features)
    model.fit(X_train1,y_train1)
    y_train_pred = model.predict(X_train1)
    y_val_pred = model.predict(X_val)
    train_score.append(accuracy_score(y_train1,y_train_pred))
    val_score.append(accuracy_score(y_val,y_val_pred))
    print('number of features per tree : {}'.format(features))
    print('train score : {}'.format(accuracy_score(y_train1,y_train_pred)))
    print('test score : {}'.format(accuracy_score(y_val,y_val_pred)))
    
plt.plot(max_features_option,train_score, label='train score')
plt.plot(max_features_option,val_score, label='val score')
plt.xlabel('max_features option')
plt.ylabel('accuracy score')
plt.legend()
plt.show()


model = RandomForestClassifier(n_estimators=200,max_depth=9, max_features= 2)

model.fit(X_train1,y_train1)
y_pred = model.predict(X_train1)
y_val_pred = model.predict(X_val)
print('Train Score : {}'.format(accuracy_score(y_train1,y_pred)))
print('validation Score : {}'.format(accuracy_score(y_val,y_val_pred)))
cross_val_score = cross_val_score(model,X_train,y_train, scoring='accuracy')
cross_val_score.mean()
