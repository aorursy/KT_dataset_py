# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resources = pd.read_csv('../input/resources.csv')

train.head()
print(train.isnull().sum())
# nearly 85% of the projects are approved
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['project_is_approved'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('project approved Status')
ax[0].set_ylabel('')

sns.countplot('project_is_approved', data=train, ax=ax[1])
ax[1].set_title('project approved Status')
plt.show()
train.groupby(['project_is_approved', 'teacher_number_of_previously_posted_projects'])['teacher_number_of_previously_posted_projects'].count()
f, ax = plt.subplots(1, 2, figsize=(20,8))
train[['teacher_number_of_previously_posted_projects', 'project_is_approved']].groupby(['teacher_number_of_previously_posted_projects']).mean().plot.bar(ax=ax[0])
ax[0].set_title('CASE_VAL vs YEAR')
sns.countplot('teacher_number_of_previously_posted_projects', hue='project_is_approved', data=train, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
#This suggests that as year increased the number of applications has increased and so has the rate of number of 
#successful applications
# we can vaguely see that if a teacher has past project submissions, then the chance of success increases for project
# approval. To make this more visible, we can partition the number of submissions in different buckets.
# The number of past project submissions range from 0 - 500
# I am creating 20 buckets of 25 each, with teachers with no past submissions in a separate bucket
train['PAST_SUBMISSION_RANGE'] = 0

train.loc[train['teacher_number_of_previously_posted_projects'] == 0 , 'PAST_SUBMISSION_RANGE'] = 0
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 1)&(train['teacher_number_of_previously_posted_projects'] < 25), 'PAST_SUBMISSION_RANGE'] = 1
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 25)&(train['teacher_number_of_previously_posted_projects'] < 50), 'PAST_SUBMISSION_RANGE'] = 2
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 50)&(train['teacher_number_of_previously_posted_projects'] < 75), 'PAST_SUBMISSION_RANGE'] = 3
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 75)&(train['teacher_number_of_previously_posted_projects'] < 100), 'PAST_SUBMISSION_RANGE'] = 4
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 100)&(train['teacher_number_of_previously_posted_projects'] < 125), 'PAST_SUBMISSION_RANGE'] = 5
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 125)&(train['teacher_number_of_previously_posted_projects'] < 150), 'PAST_SUBMISSION_RANGE'] = 6
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 150)&(train['teacher_number_of_previously_posted_projects'] < 175), 'PAST_SUBMISSION_RANGE'] = 7
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 175)&(train['teacher_number_of_previously_posted_projects'] < 200), 'PAST_SUBMISSION_RANGE'] = 8
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 200)&(train['teacher_number_of_previously_posted_projects'] < 225), 'PAST_SUBMISSION_RANGE'] = 9
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 225)&(train['teacher_number_of_previously_posted_projects'] < 250), 'PAST_SUBMISSION_RANGE'] = 10
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 250)&(train['teacher_number_of_previously_posted_projects'] < 275), 'PAST_SUBMISSION_RANGE'] = 11
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 275)&(train['teacher_number_of_previously_posted_projects'] < 300), 'PAST_SUBMISSION_RANGE'] = 12
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 300)&(train['teacher_number_of_previously_posted_projects'] < 325), 'PAST_SUBMISSION_RANGE'] = 13
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 325)&(train['teacher_number_of_previously_posted_projects'] < 350), 'PAST_SUBMISSION_RANGE'] = 14
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 350)&(train['teacher_number_of_previously_posted_projects'] < 375), 'PAST_SUBMISSION_RANGE'] = 15
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 375)&(train['teacher_number_of_previously_posted_projects'] < 400), 'PAST_SUBMISSION_RANGE'] = 16
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 400)&(train['teacher_number_of_previously_posted_projects'] < 425), 'PAST_SUBMISSION_RANGE'] = 17
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 425)&(train['teacher_number_of_previously_posted_projects'] < 450), 'PAST_SUBMISSION_RANGE'] = 18
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 450)&(train['teacher_number_of_previously_posted_projects'] < 475), 'PAST_SUBMISSION_RANGE'] = 19
train.loc[(train['teacher_number_of_previously_posted_projects'] >= 475)&(train['teacher_number_of_previously_posted_projects'] < 500), 'PAST_SUBMISSION_RANGE'] = 20
train.loc[train['teacher_number_of_previously_posted_projects'] >= 500, 'PAST_SUBMISSION_RANGE'] = 21

f, ax = plt.subplots(1, 2, figsize=(20,8))
train[['PAST_SUBMISSION_RANGE', 'project_is_approved']].groupby(['PAST_SUBMISSION_RANGE']).mean().plot.bar(ax=ax[0])
ax[0].set_title('CASE_VAL vs YEAR')
sns.countplot('PAST_SUBMISSION_RANGE', hue='project_is_approved', data=train, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
#This suggests that as year increased the number of applications has increased and so has the rate of number of 
#successful applications
# we can see from below that the chances of success increase significantly if the number of prior submissions is high, meaning the
# submitter is much well versed in what is required to get the project cleared
# Visualize
f, ax = plt.subplots(1, 2, figsize=(20,8))
train[['project_is_approved', 'teacher_prefix']].groupby(['teacher_prefix']).mean().plot.bar(ax=ax[0])
ax[0].set_title('project_is_approved vs teacher_prefix')
sns.countplot('teacher_prefix', hue='project_is_approved', data=train, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
train.groupby(['project_is_approved', 'teacher_prefix'])['teacher_prefix'].count()
# it appears that roughly all initials have rejection rate from 13-20% with lowest for Mrs and maximum for Dr.
train['Initial'] = 0

train.loc[train['teacher_prefix'] == 'Dr.' , 'Initial'] = 0
train.loc[train['teacher_prefix'] == 'Mr.' , 'Initial'] = 1
train.loc[train['teacher_prefix'] == 'Mrs.' , 'Initial'] = 2
train.loc[train['teacher_prefix'] == 'Ms.' , 'Initial'] = 3
train.loc[train['teacher_prefix'] == 'Teacher' , 'Initial'] = 4

# Visualize
f, ax = plt.subplots(1, 2, figsize=(20,8))
train[['project_is_approved', 'Initial']].groupby(['Initial']).mean().plot.bar(ax=ax[0])
ax[0].set_title('project_is_approved vs teacher_prefix')
sns.countplot('Initial', hue='project_is_approved', data=train, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
resources.head()
#resources.shape
# ensuring if each teacher has submitted single project
train.groupby(['id', 'teacher_id'])['teacher_id'].size().nlargest(10)
# some states have high success rate
train.groupby(['project_is_approved', 'school_state'])['school_state'].count()
# assign numeric value to states

# assigning numeric values to SOC with same name
state_code = []
dict = {}
code = 0
for i in range(len(train.school_state)):
    if(train.school_state[i] in dict):
        state_code.append(dict.get(train.school_state[i]))
    else:
        code += 1
        dict[train.school_state[i]] = code
        state_code.append(dict.get(train.school_state[i]))

train['STATE_CODE'] = state_code

train.head()
train.groupby(['project_is_approved', 'STATE_CODE'])['STATE_CODE'].size().nlargest(10)
f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('STATE_CODE', 'project_is_approved', data=train, ax=ax[0])
ax[0].set_title('STATE_CODE vs project_is_approved')
sns.factorplot('STATE_CODE', 'project_is_approved', data=train, ax=ax[1])
ax[1].set_title('STATE_CODE vs project_is_approved')
plt.close(2)
plt.show()
# project grade category
train.groupby(['project_is_approved', 'project_grade_category'])['project_grade_category'].count()
train['GRADE_CATEGORY'] = 0

train.loc[train['project_grade_category'] == 'Grades 3-5' , 'GRADE_CATEGORY'] = 0
train.loc[train['project_grade_category'] == 'Grades 6-8' , 'GRADE_CATEGORY'] = 1
train.loc[train['project_grade_category'] == 'Grades 9-12' , 'GRADE_CATEGORY'] = 2
train.loc[train['project_grade_category'] == 'Grades PreK-2' , 'GRADE_CATEGORY'] = 3
f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('GRADE_CATEGORY', 'project_is_approved', data=train, ax=ax[0])
ax[0].set_title('GRADE_CATEGORY vs project_is_approved')
sns.factorplot('GRADE_CATEGORY', 'project_is_approved', data=train, ax=ax[1])
ax[1].set_title('GRADE_CATEGORY vs project_is_approved')
plt.close(2)
plt.show()
train.groupby(['project_is_approved', 'project_subject_categories'])['project_subject_categories'].count()
# assign numeric value to states

# assigning numeric values to SOC with same name
category_code = []
dict = {}
code = 0
for i in range(len(train.project_subject_categories)):
    if(train.project_subject_categories[i] in dict):
        category_code.append(dict.get(train.project_subject_categories[i]))
    else:
        code += 1
        dict[train.project_subject_categories[i]] = code
        category_code.append(dict.get(train.project_subject_categories[i]))

train['CATEGORY_CODE'] = category_code

f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('CATEGORY_CODE', 'project_is_approved', data=train, ax=ax[0])
ax[0].set_title('CATEGORY_CODE vs project_is_approved')
sns.factorplot('CATEGORY_CODE', 'project_is_approved', data=train, ax=ax[1])
ax[1].set_title('CATEGORY_CODE vs project_is_approved')
plt.close(2)
plt.show()
train = train.join(resources.set_index('id'), on='id')
train['NET_PRICE'] = train['quantity'] * train['price']
train.head()
f, ax=plt.subplots(1, 2, figsize=(20, 8))
sns.distplot(train.NET_PRICE, ax=ax[0])
ax[0].set_title('NET_PRICE')
plt.show()
list(train.columns.values)
# since we see that net price is a Gaussian distribution, we can normalize it
from sklearn.preprocessing import normalize
norm = normalize(train.NET_PRICE.reshape(-1, 1), axis=0).ravel().reshape(-1, 1)
train['NORM_NET_PRICE'] = norm
train.head()
train['NORM_NET_PRICE'] = pd.cut(train.NET_PRICE, bins=15, labels=False)

bin = [];
mean = train['NET_PRICE'].mean()
std = train.loc[:, 'NET_PRICE'].std(axis=0)
min_val = train.loc[:, 'NET_PRICE'].min(axis=0)
max_val = train.loc[:, 'NET_PRICE'].max(axis=0)



bins = [min_val, mean, mean + std, mean + 2*std, mean + 3*std, max_val]

print(std)
print(bins)

train['NORM_NET_PRICE'] = pd.cut(train.NET_PRICE, bins=bins, labels=False)
train.head()
train.groupby(['NORM_NET_PRICE'])['NORM_NET_PRICE'].count()
f, ax=plt.subplots(1, 2, figsize=(20, 8))
sns.countplot(train.NORM_NET_PRICE, ax=ax[0])
ax[0].set_title('NORM_NET_PRICE')
plt.show()
train.drop(['teacher_id', 'teacher_prefix', 'school_state', 'project_submitted_datetime', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary', 'teacher_number_of_previously_posted_projects', 'description', 'quantity', 'price', 'NET_PRICE'], axis=1, inplace=True)
train.shape
train.head()
train.isnull().sum()
# new value
train.loc[(train.NORM_NET_PRICE.isnull())] = 5.0
train.drop(['id'], axis=1, inplace=True)
# heatmap
sns.heatmap(train.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18, 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# since i see high correlation between variables, I prefer using a linear regression model
############# Prediction #################
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

y = train.project_is_approved;
#y = y.reshape([-1, 1])
X = train.drop('project_is_approved', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train.isnull().sum()
# declare data preprocessing steps 
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
#pipeline = make_pipeline(preprocessing.StandardScaler(), 
#                          DecisionTreeClassifier(max_depth=5))

# declare hyperparameters to tun
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth' : [None, 5, 3, 1]}

# Tune model using cross-validation pipeline, 10 fold
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#clf = GridSearchCV(pipeline, {}, cv=10)
clf.fit(X_train, y_train)

# evaluate model pipeline on test data
pred = clf.predict(X_test)
print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))