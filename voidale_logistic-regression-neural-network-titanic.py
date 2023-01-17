# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebrxa
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection, ensemble
from scipy.stats import skew
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
combined = [train_data, test_data]

# show column headers
print(train_data.columns.values, test_data.columns.values)
# print example
train_data.head()
test_data.head()
# stats about the data
train_data.describe()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# drop unneeded features
train_data = train_data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

# train data skwed
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index

skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

train_data[skewed_feats] = np.log1p(train_data[skewed_feats])

# test data skwed
numeric_feats = test_data.dtypes[test_data.dtypes != "object"].index

skewed_feats = test_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

test_data[skewed_feats] = np.log1p(test_data[skewed_feats])

combine = [train_data, test_data]
# Complete data
freq_port = train_data.Embarked.dropna().mode()[0]

for dataset in combine:
    # complete fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
    # use popular port for empty
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
    # complete age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

print (train_data.shape, test_data.shape)
print (train_data.isnull().sum(), test_data.isnull().sum())


## Create more data
for dataset in combine:
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (train_data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
train_data['Title'] = train_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# print(train_data['Title'].value_counts())


#preview data again
train_data.head()
# convert using label encoder labels to numbers

#code categorical data
label = LabelEncoder()
for dataset in combine:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    
#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')

#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(train_data[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')


data1_dummy.head()

# double check for nulls
print (train_data.isnull().sum(), test_data.isnull().sum())
train_data.head()
#split train and test data with function defaults
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train_data[data1_x_calc], train_data[Target], random_state = 0, test_size = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train_data[data1_x_bin], train_data[Target], random_state = 0, test_size = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], train_data[Target], random_state = 0, test_size = 0)


print("Data Shape: {}".format(train_data.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))
print("Train1_y Shape: {}".format(train1_y.shape))
print("Test1_y Shape: {}".format(test1_y.shape))

# alg = ensemble.RandomForestClassifier(n_estimators = 100)
# # alg = LogisticRegression()
# alg.fit()
# cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 
# cv_results = model_selection.cross_validate(alg, train_data[data1_x_bin], train_data[Target], cv  = cv_split)
# print(cv_results['fit_time'].mean(), cv_results['train_score'].mean(), cv_results['test_score'].mean(), cv_results['test_score'].min())

# Logistic Regression
X_train = train_data[data1_x_calc]
Y_train = train_data[Target]
X_test  = test_data[data1_x_calc]

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

raf = ensemble.RandomForestClassifier(n_estimators = 100)
raf.fit(X_train, Y_train)
Y_pred2 = raf.predict(X_test)
acc_log2 = round(raf.score(X_train, Y_train) * 100, 2)

print ('Logistic Regression: ', acc_log, 'Random Forest: ', acc_log2)

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, y)
# random_forest_preds = random_forest.predict(X_test)
# random_forest.score(X_train, y)
# accuracy = round(random_forest.score(X_train, y) * 100, 2)
# print(accuracy)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred2
    })
submission.to_csv('random_forest_y_pred_2_f.csv', index=False)
print('done')

