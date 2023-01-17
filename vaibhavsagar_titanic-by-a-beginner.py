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
# Used to display the full output without the ...
pd.set_option('display.expand_frame_repr', False)

# Ignore warnings thrown by Seaborn
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
# Import charting libraries
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")
validation = pd.read_csv("../input/test.csv")
# Check the train dataset.
print(train.columns.values)
train.describe(include='all')
# Understand the datatypes
print(train.dtypes)
print()
# Focus first on null values
print(train.isna().sum())
# Check the validation dataset also.
print(train.columns.values)
train.describe(include='all')
print(validation.dtypes)
print()
print(validation.isna().sum())
# Check the correlation for the current numeric feature set.
print(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())
sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# List the features again
print(train.columns.values)
# Lets see the relation between Pclass and Survived
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Fare")
group = pd.cut(train.Fare, [0,50,100,150,200,550])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Fare', aggfunc='count')
piv_fare.plot(kind='bar')
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
group = pd.cut(train.Age, [0,14,30,60,100])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Age', aggfunc='count')
piv_fare.plot(kind='bar')
print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x='Embarked', y='Survived',  kind='bar', data=train)
sns.catplot('Pclass', kind='count', col='Embarked', data=train)
print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')
print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=train, kind='bar')
# Explanation for the next code block used to get the Titles
# Using this output to explain the string manipulation below
print(train.Name.head(1))
print()
# The above returns a single name for e.g. Braund, Mr. Owen Harris.
# Calling str returns a String object
print(train.Name.head(1).str)
print()
# Next we split the string into a List with a comma as the separator
print(train.Name.head(1).str.split(','))
print()
# Similary we remove the . and then strip the remaining string to get the title.
# We pick the second item of the
print(train.Name.head(1).str.split(',').str[1])
print()
# Get the titles
for dataset in [train, validation]:
    # Use split to get only the titles from the name
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(dataset['Title'].value_counts())
    print()
sns.catplot(x='Survived', y='Title', data=train, kind ='bar')
for df in [train, validation]:
    print(df.shape)
    print()
    print(df.isna().sum())
# Drop rows with nulls for Embarked
for df in [train, validation]:
    df.dropna(subset = ['Embarked'], inplace = True)
print(train[train['Fare'].isnull()])
print() 
# 1 row with null Fare in validation
print(validation[validation['Fare'].isnull()])
# We can deduce that Pclass should be related to Fares.
sns.catplot(x='Pclass', y='Fare', data=validation, kind='point')
# There is a clear relation between Pclass and Fare. We can use this information to impute the missing fare value.
# We see that the passenger is from Pclass 3. So we take a median value for all the Pclass 3 fares.
validation['Fare'].fillna(validation[validation['Pclass'] == 3].Fare.median(), inplace = True)
print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x='Age', y='Title', data=train, kind ='bar')
# Returns titles from the passed in series.
def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
# Prints the count of titles with nulls for the train dataframe.
print(getTitle(train[train.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = train['Title'] == 'Mr'
miss_mask = train['Title'] == 'Miss'
mrs_mask = train['Title'] == 'Mrs'
master_mask = train['Title'] == 'Master'
dr_mask = train['Title'] == 'Dr'
train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())
train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())
train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())
train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())
train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())
# Prints the count of titles with nulls for the train dataframe. -- Should be empty this time.
print()
print(getTitle(train[train.Age.isnull()].Name).value_counts())
# Prints the count of titles with nulls for the validation dataframe.
print(getTitle(validation[validation.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = validation['Title'] == 'Mr'
miss_mask = validation['Title'] == 'Miss'
mrs_mask = validation['Title'] == 'Mrs'
master_mask = validation['Title'] == 'Master'
ms_mask = validation['Title'] == 'Ms'
validation.loc[mr_mask, 'Age'] = validation.loc[mr_mask, 'Age'].fillna(validation[validation.Title == 'Mr'].Age.mean())
validation.loc[miss_mask, 'Age'] = validation.loc[miss_mask, 'Age'].fillna(validation[validation.Title == 'Miss'].Age.mean())
validation.loc[mrs_mask, 'Age'] = validation.loc[mrs_mask, 'Age'].fillna(validation[validation.Title == 'Mrs'].Age.mean())
validation.loc[master_mask, 'Age'] = validation.loc[master_mask, 'Age'].fillna(validation[validation.Title == 'Master'].Age.mean())
validation.loc[ms_mask, 'Age'] = validation.loc[ms_mask, 'Age'].fillna(validation[validation.Title == 'Miss'].Age.mean())
# Prints the count of titles with nulls for the validation dataframe. -- Should be empty this time.
print(getTitle(validation[validation.Age.isnull()].Name).value_counts())
# train.Age.fillna(train.Age.median(), inplace=True)
# validation.Age.fillna(validation.Age.median(), inplace=True)
print(train.isna().sum())
print(validation.isna().sum())
train.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [train, validation]]
[train, validation] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, validation]]
for df in [train, validation]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)
[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, validation]]
# We see that there are a few non standard titles. Some of them are just French titles
# with the same meaning as in English while others point to people who would probably
# have more privileges or military training etc and can be placed in a separate category.
# French titles - https://en.wikipedia.org/wiki/French_honorifics
# Mlle - https://en.wikipedia.org/wiki/Mademoiselle_(title)
# Mme - https://en.wikipedia.org/wiki/Madam
# Mme was a bit harder to understand as Wikipedia says that its used for adult women
# but doesn't given any pointers towards their marital status.
# Searching up on Google and considering that the title is used for adult women
# we can assume that this title was usually assigned to married women.
# https://www.frenchtoday.com/blog/french-culture/madame-or-mademoiselle-a-delicate-question
# Ms - An alternate abbrevation for Miss
train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
validation['Title'] = validation['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [train, validation]]
[train, validation] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, validation]]
# Check the updated dataset
print(train.columns.values)
print(validation.columns.values)
# Check the correlation with the updated datasets
train.corr()
# Split the the dataset into train and test sets.
from sklearn.model_selection import train_test_split
# Use only the features with a coeefficient greater than 0.3
X = train[['Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Embarked_C',
       'Embarked_S', 'HasCabin', 'FamilySize', 'Title_Master', 'Title_Mr',
       'Title_Mrs', 'Title_Special']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)
# We will also create a base model to check the goodness of our model.
# First we see the actual number of survivors
print(y.value_counts())
# We will select the larger number and consider that everyone dies to create a baseline.
y_default = pd.Series([0] * train['Survived'].shape[0], name = 'Survived')
print(y_default.value_counts())
# Calculate the baseline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y, y_default))
print()
print(accuracy_score(y, y_default))
# So if we assumed that everyone dies we would be correct 61% of the time.
# So this is the bare minimun level of accuracy our prediciton should aim to improve upon.
# First attempt with LinearSVC
from sklearn.svm import LinearSVC
# Looking into the documentation points us to set dual=False for cases with n_samples > n_features.
classifier = LinearSVC(dual=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
# Next up we will try KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
# KNN isn't useful for us so we now move to a few popular ensemble estimators
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
print("AdaBoostClassifier")
ada_boost_classifier = AdaBoostClassifier()
ada_boost_classifier.fit(X_train, y_train)
y_pred = ada_boost_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(ada_boost_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("BaggingClassifier")
bagging_classifier = BaggingClassifier()
bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(bagging_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("ExtraTreesClassifier")
extra_trees_classifier = ExtraTreesClassifier(n_estimators=100)
extra_trees_classifier.fit(X_train, y_train)
y_pred = extra_trees_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(extra_trees_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("GradientBoostingClassifier")
gradient_boosting_classifier = GradientBoostingClassifier()
gradient_boosting_classifier.fit(X_train, y_train)
y_pred = gradient_boosting_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(gradient_boosting_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
print("RandomForestClassifier")
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(random_forest_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
# Will also try XGB based on its popularity and relevance here.
from xgboost import XGBClassifier
xgboost_classifier = XGBClassifier()
xgboost_classifier.fit(X_train, y_train)
y_pred = xgboost_classifier.predict(X_test)
# Print the confusion matrix
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
print(confusion_matrix(y_test, y_pred))
# Print the accuracy score
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
print(accuracy_score(y_test, y_pred))
scores = cross_val_score(xgboost_classifier, X_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
# X = train.iloc[:, 1:]
# y = train.iloc[:, 0]
# print(X.columns.values)
# xgboost_classifier = XGBClassifier()
# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=xgboost_classifier, cv=10, scoring='accuracy')
# rfecv = rfecv.fit(X, y)
# print(X.columns[rfecv.support_])
# Now we will pass the validation set provided for creating our submission
# Pick the same columns as in X_test
X_validation = validation[['Age', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Embarked_C',
       'Embarked_S', 'HasCabin', 'FamilySize', 'Title_Master', 'Title_Mr',
       'Title_Mrs', 'Title_Special']]
# Call the predict from the created classifier
y_valid = xgboost_classifier.predict(X_validation)
print(validation.columns.values)
# Creating final output file
validation_pId = validation.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})
print(my_submission['Survived'].value_counts())
my_submission.to_csv('submission.csv', index = False)