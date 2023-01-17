#learning from https://www.kaggle.com/startupsci/titanic-data-science-solutions

# Imports
import pandas as pd 
import numpy as np

# Standardize, Label Encode, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer

# Model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor

import seaborn as sns
import matplotlib.pyplot as plt

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


train_source = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_source = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

all_data = [train_source,test_source]

############################
# Make new feature 'title' #
############################

# extract title from name
title_list_train = train_source.Name.str.extract('([A-Za-z]+)\.', expand=False)
train_source['title'] = title_list_train

title_list_test = test_source.Name.str.extract('([A-Za-z]+)\.', expand=False)
test_source['title'] = title_list_test

# change some to Miss and Mrs
train_source['title'] = train_source['title'].replace('Mlle','Miss')
train_source['title'] = train_source['title'].replace('Ms','Miss')
train_source['title'] = train_source['title'].replace('Mme','Mrs')

test_source['title'] = test_source['title'].replace('Mlle','Miss')
test_source['title'] = test_source['title'].replace('Ms','Miss')
test_source['title'] = test_source['title'].replace('Mme','Mrs')

# combine title with small counts to 'Other'
small_title = train_source.title.value_counts().index[train_source.title.value_counts() < 5]

train_source['title'] = train_source['title'].replace(small_title,'Other')
test_source['title'] = test_source['title'].replace(small_title,'Other')

train_source['title'] = train_source['title'].replace('Dr','Other')
test_source['title'] = test_source['title'].replace('Dr','Other')

test_source['title'] = test_source['title'].replace('Dona','Other')


####################
# fill missing age #
####################

# get median age of the 3 class (general)
md_age_class1 = train_source.loc[(train_source.Pclass == 1),:].Age.dropna().median()
md_age_class2 = train_source.loc[(train_source.Pclass == 2),:].Age.dropna().median()
md_age_class3 = train_source.loc[(train_source.Pclass == 3),:].Age.dropna().median()

# get median age by title, class = 1
md_age_class1_mr = train_source.loc[(train_source.title == 'Mr') & (train_source.Pclass == 1),:].Age.dropna().median()
md_age_class1_mrs = train_source.loc[(train_source.title == 'Mrs') & (train_source.Pclass == 1),:].Age.dropna().median()
md_age_class1_miss = train_source.loc[(train_source.title == 'Miss') & (train_source.Pclass == 1),:].Age.dropna().median()
md_age_class1_master = train_source.loc[(train_source.title == 'Master') & (train_source.Pclass == 1),:].Age.dropna().median()

# get median age by title, class = 2
md_age_class2_mr = train_source.loc[(train_source.title == 'Mr') & (train_source.Pclass == 2),:].Age.dropna().median()
md_age_class2_mrs = train_source.loc[(train_source.title == 'Mrs') & (train_source.Pclass == 2),:].Age.dropna().median()
md_age_class2_miss = train_source.loc[(train_source.title == 'Miss') & (train_source.Pclass == 2),:].Age.dropna().median()
md_age_class2_master = train_source.loc[(train_source.title == 'Master') & (train_source.Pclass == 2),:].Age.dropna().median()

# get median age by title, class = 3
md_age_class3_mr = train_source.loc[(train_source.title == 'Mr') & (train_source.Pclass == 3),:].Age.dropna().median()
md_age_class3_mrs = train_source.loc[(train_source.title == 'Mrs') & (train_source.Pclass == 3),:].Age.dropna().median()
md_age_class3_miss = train_source.loc[(train_source.title == 'Miss') & (train_source.Pclass == 3),:].Age.dropna().median()
md_age_class3_master = train_source.loc[(train_source.title == 'Master') & (train_source.Pclass == 3),:].Age.dropna().median()

for data in all_data:
    # fill age na class 1
    data.loc[(data.title == 'Mr') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Mr') & (data.Pclass == 1),'Age'].fillna(md_age_class1_mr)
    data.loc[(data.title == 'Mrs') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Mrs') & (data.Pclass == 1),'Age'].fillna(md_age_class1_mrs)
    data.loc[(data.title == 'Miss') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Miss') & (data.Pclass == 1),'Age'].fillna(md_age_class1_miss)
    data.loc[(data.title == 'Master') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Master') & (data.Pclass == 1),'Age'].fillna(md_age_class1_master)
    data.loc[(data.title == 'Rev') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Rev') & (data.Pclass == 1),'Age'].fillna(md_age_class1)
    data.loc[(data.title == 'Other') & (data.Pclass == 1),'Age'] = data.loc[(data.title == 'Other') & (data.Pclass == 1),'Age'].fillna(md_age_class1)
    
    # fill age na class 2
    data.loc[(data.title == 'Mr') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Mr') & (data.Pclass == 2),'Age'].fillna(md_age_class2_mr)
    data.loc[(data.title == 'Mrs') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Mrs') & (data.Pclass == 2),'Age'].fillna(md_age_class2_mrs)
    data.loc[(data.title == 'Miss') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Miss') & (data.Pclass == 2),'Age'].fillna(md_age_class2_miss)
    data.loc[(data.title == 'Master') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Master') & (data.Pclass == 2),'Age'].fillna(md_age_class2_master)
    data.loc[(data.title == 'Rev') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Rev') & (data.Pclass == 2),'Age'].fillna(md_age_class2)
    data.loc[(data.title == 'Other') & (data.Pclass == 2),'Age'] = data.loc[(data.title == 'Other') & (data.Pclass == 2),'Age'].fillna(md_age_class2)

    # fill age na class 3
    data.loc[(data.title == 'Mr') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Mr') & (data.Pclass == 3),'Age'].fillna(md_age_class3_mr)
    data.loc[(data.title == 'Mrs') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Mrs') & (data.Pclass == 3),'Age'].fillna(md_age_class3_mrs)
    data.loc[(data.title == 'Miss') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Miss') & (data.Pclass == 3),'Age'].fillna(md_age_class3_miss)
    data.loc[(data.title == 'Master') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Master') & (data.Pclass == 3),'Age'].fillna(md_age_class3_master)
    data.loc[(data.title == 'Rev') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Rev') & (data.Pclass == 3),'Age'].fillna(md_age_class3)
    data.loc[(data.title == 'Other') & (data.Pclass == 3),'Age'] = data.loc[(data.title == 'Other') & (data.Pclass == 3),'Age'].fillna(md_age_class3)

    
############################################
# fill missing emarked on train - Mode 'S' #
############################################
train_source.loc[62,'Embarked'] = 'S'
train_source.loc[830,'Embarked'] = 'S'


#############################################################
# fill missing fare on test with "mean of fare in Pclass 3" #
#############################################################
mean_fare_class3 = train_source.loc[train_source.Pclass == 3].Fare.mean()
test_source.loc[1044,'Fare'] = mean_fare_class3    


##############################
# new feature: family member #
##############################
train_source['family'] = train_source.SibSp + train_source.Parch
test_source['family'] = test_source.SibSp + test_source.Parch

#has family = 1, no family = 0
train_source['hasFamily'] = 0
train_source.loc[train_source['family'] > 0, 'hasFamily'] = 1

test_source['hasFamily'] = 0
test_source.loc[test_source['family'] > 0, 'hasFamily'] = 1


#############
# age group #
#############
train_source.loc[train_source['Age'] <= 16, 'Age'] = 0
train_source.loc[(train_source['Age'] > 16) & (train_source['Age'] <= 32), 'Age'] = 1
train_source.loc[(train_source['Age'] > 32) & (train_source['Age'] <= 48), 'Age'] = 2
train_source.loc[(train_source['Age'] > 48) & (train_source['Age'] <= 64), 'Age'] = 3
train_source.loc[(train_source['Age'] > 64), 'Age'] = 4

test_source.loc[test_source['Age'] <= 16, 'Age'] = 0
test_source.loc[(test_source['Age'] > 16) & (test_source['Age'] <= 32), 'Age'] = 1
test_source.loc[(test_source['Age'] > 32) & (test_source['Age'] <= 48), 'Age'] = 2
test_source.loc[(test_source['Age'] > 48) & (test_source['Age'] <= 64), 'Age'] = 3
test_source.loc[(test_source['Age'] > 64), 'Age'] = 4


###########################
# age * class interaction #
###########################
# train_source['Age_Class'] = train_source.Age * train_source.Pclass
# test_source['Age_Class'] = test_source.Age * test_source.Pclass


##########################################
# drop columns with litte/no information #
# drop 'Sibsp', 'Parch'                  #
##########################################
col_drop = ['Name','Ticket','Cabin', 'SibSp', 'Parch','family']

train_source = train_source.drop(col_drop,axis=1)
test_source = test_source.drop(col_drop,axis=1)

##########    
# set up #
##########

y = train_source.Survived

train = train_source.drop('Survived', axis=1)
test = test_source

print('missing in train:', train.isnull().sum())
print('missing in test:', test.isnull().sum())

train.head()
#test.head()
test_x = test

train_x,valid_x,train_y,valid_y = train_test_split(train, y, test_size = 0.3, random_state = 0)

print('train x:', train_x.shape)
print('train y:', train_y.shape)
print('valid x:', valid_x.shape)
print('valid y:', valid_y.shape)
print('test x:', test_x.shape)
# log Fare
train_x.Fare = np.log1p(train_x.Fare)
valid_x.Fare = np.log1p(valid_x.Fare)
test_x.Fare = np.log1p(test_x.Fare)
#impute "Embarked"
ec = LabelEncoder()
train_x['Sex'] = ec.fit_transform(train_x['Sex'])
valid_x['Sex'] = ec.transform(valid_x['Sex'])
test_x['Sex'] = ec.transform(test_x['Sex'])

train_x['Embarked'] = ec.fit_transform(train_x['Embarked'])
valid_x['Embarked'] = ec.transform(valid_x['Embarked'])
test_x['Embarked'] = ec.transform(test_x['Embarked'])

train_x['title'] = ec.fit_transform(train_x['title'])
valid_x['title'] = ec.transform(valid_x['title'])
test_x['title'] = ec.transform(test_x['title'])
# SVC
model = svm.SVC()
model.fit(train_x, train_y)

print('error on train:', model.score(train_x,train_y))
print('error on train:', model.score(valid_x,valid_y))
# KNN
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(train_x, train_y)

print('error on train:', model.score(train_x,train_y))
print('error on train:', model.score(valid_x,valid_y))
# logistic regression 
# best so far on leaderboard
model = LogisticRegression()
model.fit(train_x, train_y)

print('error on train:', model.score(train_x,train_y))
print('error on train:', model.score(valid_x,valid_y))
# random forest
model = RandomForestClassifier(random_state=1)
model.fit(train_x, train_y)

print('error on train:', model.score(train_x,train_y))
print('error on train:', model.score(valid_x,valid_y))
# decision tree
model = DecisionTreeClassifier()
model.fit(train_x, train_y)

print('error on train:', model.score(train_x,train_y))
print('error on train:', model.score(valid_x,valid_y))
# prediction 
pred = model.predict(test_x)
pred
submission = pd.DataFrame({'PassengerID': test_source.index,'Survived': pred})
submission.to_csv('submission.csv',index=False)