# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/h1b_kaggle.csv')
data.head()
data.isnull().sum()
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['CASE_STATUS'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Case Status')
ax[0].set_ylabel('')

sns.countplot('CASE_STATUS', data=data, ax=ax[1])
ax[1].set_title('Case Status')
plt.show()
sns.countplot('CASE_STATUS', data=data)
fig = plt.gcf()
fig.set_size_inches(30, 8)
plt.show()
data.groupby(['CASE_STATUS', 'FULL_TIME_POSITION'])['FULL_TIME_POSITION'].count()
data['CASE_VAL'] = 0
# assign NAN values with mean values based on initial
data.loc[(data['CASE_STATUS'] == 'CERTIFIED'), 'CASE_VAL'] = 0
data.loc[(data['CASE_STATUS'] == 'CERTIFIED-WITHDRAWN'), 'CASE_VAL'] = 1
data.loc[(data['CASE_STATUS'] == 'DENIED'), 'CASE_VAL'] = 2
data.loc[(data['CASE_STATUS'] == 'INVALIDATED'), 'CASE_VAL'] = 3
data.loc[(data['CASE_STATUS'] == 'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED'), 'CASE_VAL'] = 4
data.loc[(data['CASE_STATUS'] == 'WITHDRAWN'), 'CASE_VAL'] = 5
data.loc[data.CASE_STATUS.isnull()] = 5
sns.countplot('CASE_VAL', data=data)
fig = plt.gcf()
fig.set_size_inches(30, 8)
plt.show()
f, ax = plt.subplots(1, 2, figsize=(20,8))
data[['YEAR', 'CASE_VAL']].groupby(['YEAR']).mean().plot.bar(ax=ax[0])
ax[0].set_title('CASE_VAL vs YEAR')
sns.countplot('YEAR', hue='CASE_VAL', data=data, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
#This suggests that as year increased the number of applications has increased and so has the rate of number of 
#successful applications
data['WAGE_RANGE'] = 0

data.loc[data.PREVAILING_WAGE.isnull()] = 0
data.loc[data['PREVAILING_WAGE'] < 50000 , 'WAGE_RANGE'] = 0
data.loc[(data['PREVAILING_WAGE'] >= 50000)&(data['PREVAILING_WAGE'] < 100000), 'WAGE_RANGE'] = 1
data.loc[(data['PREVAILING_WAGE'] >= 100000)&(data['PREVAILING_WAGE'] < 150000), 'WAGE_RANGE'] = 2
data.loc[(data['PREVAILING_WAGE'] >= 150000)&(data['PREVAILING_WAGE'] < 200000), 'WAGE_RANGE'] = 3
data.loc[(data['PREVAILING_WAGE'] >= 200000)&(data['PREVAILING_WAGE'] < 250000), 'WAGE_RANGE'] = 4
data.loc[(data['PREVAILING_WAGE'] >= 250000)&(data['PREVAILING_WAGE'] < 300000), 'WAGE_RANGE'] = 5
data.loc[data['PREVAILING_WAGE'] >= 300000, 'WAGE_RANGE'] = 6

data.head()
f, ax = plt.subplots(1, 2, figsize=(20,8))
data[['WAGE_RANGE', 'CASE_VAL']].groupby(['WAGE_RANGE']).mean().plot.bar(ax=ax[0])
ax[0].set_title('CASE_VAL vs WAGE_RANGE')
sns.countplot('WAGE_RANGE', hue='CASE_VAL', data=data, ax=ax[1])
ax[1].set_title('cwc')
plt.show()
#We observe that maximum applications fall in WAGE_RANGE 50k - 100k and hence has maximum accepts. 
#The percentage of accepts for this range is way higher than any other wage range. An applicant
#in this range has more chances of getting accepted.

# separate State information from Worksite
data['STATE'] = data.WORKSITE.str.split(',').str[1]
print(data.head())
data.groupby(['CASE_VAL', 'STATE'])['STATE'].count()
#We observe that some states have very high percentage of accepts vs rejects
# similarly let's see the impact of SOC_NAME
data.groupby(['CASE_VAL', 'SOC_NAME'])['SOC_NAME'].count()
#SOC_NAME like accountants and auditors have a high chance of acceptance
# observe the employers with largest counts
data.groupby(['CASE_VAL', 'EMPLOYER_NAME'])['EMPLOYER_NAME'].size().nlargest(50)
# assigning numeric values to employers with same name
employer_code = []
dict = {}
code = 0
for i in range(len(data.EMPLOYER_NAME)):
    if(data.EMPLOYER_NAME[i] in dict):
        employer_code.append(dict.get(data.EMPLOYER_NAME[i]))
    else:
        code += 1
        dict[data.EMPLOYER_NAME[i]] = code
        employer_code.append(dict.get(data.EMPLOYER_NAME[i]))

data['EMPLOYER_CODE'] = employer_code       
#data.head(2)
# assigning numeric values to SOC with same name
soc_code = []
dict = {}
code = 0
for i in range(len(data.SOC_NAME)):
    if(data.SOC_NAME[i] in dict):
        soc_code.append(dict.get(data.SOC_NAME[i]))
    else:
        code += 1
        dict[data.SOC_NAME[i]] = code
        soc_code.append(dict.get(data.SOC_NAME[i]))

data['SOC_CODE'] = soc_code       
#data.head(2)
data.head()
# normalizing year value
data['YEAR_CODE'] = 0
# assign NAN values with mean values based on initial
data.loc[(data['YEAR'] == 2011), 'YEAR_CODE'] = 1
data.loc[(data['YEAR'] == 2012), 'YEAR_CODE'] = 2
data.loc[(data['YEAR'] == 2013), 'YEAR_CODE'] = 3
data.loc[(data['YEAR'] == 2014), 'YEAR_CODE'] = 4
data.loc[(data['YEAR'] == 2015), 'YEAR_CODE'] = 5
data.loc[(data['YEAR'] == 2016), 'YEAR_CODE'] = 6
# I have selected columns CASE_VAL, SOC_CODE, EMPLOYER_CODE, WAGE_RANGE,YEAR
# Need to convert State to numeric values as well
# I am letting go of job title for now
# distributing employers in 5 bins, 5 being highest for Infosys Limited
state_code = []
dict = {}
code = 0
for i in range(len(data.STATE)):
    if(data.STATE[i] in dict):
        state_code.append(dict.get(data.STATE[i]))
    else:
        code += 1
        dict[data.STATE[i]] = code
        state_code.append(dict.get(data.STATE[i]))

data['STATE_CODE'] = state_code       
#data.head(2)
data.head()
data['FULL_TIME_CODE'] = 0
# assign NAN values with mean values based on initial
data.loc[(data['FULL_TIME_POSITION'] == 'Y'), 'FULL_TIME_CODE'] = 0
data.loc[(data['FULL_TIME_POSITION'] == 'N'), 'FULL_TIME_CODE'] = 1
data.drop(['Unnamed: 0', 'YEAR', 'CASE_STATUS', 'EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE', 'lon', 'lat', 'STATE'], axis=1, inplace=True)
data.shape
data.head()
# heatm
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18, 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
############# Prediction #################
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

y = data.CASE_VAL;
#y = y.reshape([-1, 1])
X = data.drop('CASE_VAL', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# declare data preprocessing steps 
#pipeline = make_pipeline(preprocessing.StandardScaler(), 
#                         RandomForestRegressor(n_estimators=100))
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                          LogisticRegression())

# declare hyperparameters to tun
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth' : [None, 5, 3, 1]}

# Tune model using cross-validation pipeline, 10 fold
#clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf = GridSearchCV(pipeline, {}, cv=10)
clf.fit(X_train, y_train)

# evaluate model pipeline on test data
pred = clf.predict(X_test)
print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))
