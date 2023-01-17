# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import re

import sys

import random

import time

import scipy as sp

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)



from sklearn.svm import SVC

from sklearn.model_selection import cross_validate

from pandas import DataFrame as df

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

from subprocess import check_output



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8





# Any results you write to the current directory are saved as output.
data_raw = pd.read_csv('../input/train.csv')

data_val = pd.read_csv('../input/test.csv')

PassengerId = data_val['PassengerId']

# combine train and test

data1 = data_raw.copy(deep=True)

data_cleaner = [data1, data_val]



print('Train columns with null value: \n', data1.isnull().sum())

print("-"*10)

print('Test/Validation columns with null valu:\n', data_val.isnull().sum())

print("-"*10)

data_raw.describe(include='all')
# Calculate weight of evidence

def cacl_woe(data, col, target):

    subdata = df(data.groupby(col)[col].count())

    suby = df(data.groupby(col)[target].sum())

    data = df(pd.merge(subdata, suby, how='left', left_index=True, right_index=True))

    b_total = data[target].sum()

    total = data[col].sum()

    g_total = total - b_total

    data['bad'] = data.apply(lambda x:round(x[target]/b_total, 3), axis=1)

    data['good'] = data.apply(lambda x:round((x[col]-x[target])/g_total, 3), axis=1)

    data['WOE'] = data.apply(lambda x:np.log(x.bad/x.good), axis=1)

    return data.loc[:, ["bad", "good", "WOE"]]

# calculate information value

def calc_iv(data):

    data['IV'] = data.apply(lambda x: (x.bad-x.good)*x.WOE, axis=1)

    IV = sum(data['IV'])

    return IV
for dataset in data_cleaner:

    # complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)

    # complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# delete the cabin feature/column and other previously stated to excude in train dataset

drop_column = ['PassengerId', 'Cabin', 'Ticket']

data1.drop(drop_column, axis=1, inplace=True)



print(data1.isnull().sum())

print('-'*10)

print(data_val.isnull().sum())
for dataset in data_cleaner:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0

    dataset['Title'] = dataset['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    

# clean rare title name

stat_min = 10

title_names = (data1['Title'].value_counts() < stat_min)

data1['Title']=data1['Title'].apply(lambda x:'Misc' if title_names.loc[x]==True else x)

print(data1['Title'].value_counts())

print('-'*10)



data1.info()

data_val.info()

data1.sample(10)
label = LabelEncoder()

for dataset in data_cleaner:

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])



# define y variable as target/outcome

Target = ['Survived']

# define x variable as origin feature

data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

data1_xy = Target + data1_x

print('Original X Y: ',data1_xy, '\n')



# define x variable for original bin to remove continuous variables

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

data1_xy_bin = Target + data1_x_bin

print('Bin X Y :' ,data1_xy_bin, '\n')



# define x and y  for dummy feature original

data1_dummy = pd.get_dummies(data1[data1_x])

data1_x_dummy = data1_dummy.columns.tolist()

data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y : ', data1_xy_dummy, '\n')



data1_dummy.head()
print('Train columns with null values: \n', data1.isnull().sum())

print('-'*10)

print(data1.info())

print('-'*10)



print('Test/Validation column with null values:\n', data_val.isnull().sum())

print('-'*10)

print(data_val.info())

print('-'*10)



data_raw.describe(include='all')
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state=0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state=0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target],random_state=0)



print('Data1 shape :{}'.format(data1.shape))

print('Train1 shape:{}'.format(train1_x.shape))

print('Test1 shape:{}'.format(test1_x.shape))

train1_x_bin.head()
# Variable correlation by survival using

for x in data1_x:

    if data1[x].dtype != 'float64':

        print('Survival Correlation by: ', x)

        print(data1[[x,Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')



print(pd.crosstab(data1['Title'], data1[Target[0]]))
plt.figure(figsize=[16, 12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans=True, meanline=True)

plt.title('Frare Boxplot')

plt.ylabel('Fare($)')



plt.subplot(232)

plt.boxplot(x=data1['Age'], showmeans=True, meanline=True)

plt.title('Age Boxplot')

plt.ylabel('Age(Years)')



plt.subplot(233)

plt.boxplot(x=data1['FamilySize'], showmeans=True, meanline=True)

plt.title('FamilySize Boxplot')

plt.ylabel('Famiily Size (#)')



plt.subplot(234)

plt.hist(x=[data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']],stacked=True,color=['g','r'],label=['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of passengers ')

plt.legend()



plt.subplot(235)

plt.hist(x=[data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']],stacked=True,color=['g','r'],label=['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (years)')

plt.ylabel('# of passengers ')

plt.legend()



plt.subplot(236)

plt.hist(x=[data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']],stacked=True,color=['g','r'],label=['Survived','Dead'])

plt.title('FamilySize Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of passengers ')

plt.legend()
combinedata.Embarked[combinedata.Embarked.isnull()] = combinedata.Embarked.dropna().mode().values

combinedata.info()
combinedata['Fare'] = combinedata[['Fare']].fillna(combinedata.groupby('Pclass').transform(np.mean))
combinedata['Cabin'] = combinedata.Cabin.fillna('u0')

combinedata.info()
train_data = combinedata[0:891]

train_data.info()

# fill null age value for traindata

age_df = train_data[['Age', 'Survived', 'Parch', 'Fare', 'SibSp', 'Pclass']]

age_df_notnull = age_df.loc[(train_data['Age'].notnull())]

age_df_isnull = age_df.loc[(train_data['Age'].isnull())]

X = age_df_notnull.values[:, 1:]

Y = age_df_notnull.values[:, 0]



RFR = RandomForestRegressor(n_estimators=1000, n_jobs=1)

RFR.fit(X, Y)

predictAge = RFR.predict(age_df_isnull.values[:, 1:])

train_data.loc[(train_data['Age'].isnull(), ['Age'])] = predictAge



train_data.info()

# fill none age in test

test_data = combinedata[891:]

test_data.info()

age_df_test = test_data[['Age', 'Parch', 'Fare', 'SibSp', 'Pclass']]

age_df_test_notnull = age_df_test.loc[(test_data['Age'].notnull())]

age_df_test_isnull = age_df_test.loc[(test_data['Age'].isnull())]

X_test = age_df_test_notnull.values[:, 1:]

Y_test = age_df_test_notnull.values[:, 0]



regress = RandomForestRegressor(n_estimators=1000, n_jobs=1)

regress.fit(X_test, Y_test)

predictAge = regress.predict(age_df_test_isnull.values[:, 1:])

test_data.loc[(test_data['Age'].isnull(), ['Age'])] = predictAge
combinedata = pd.concat([train_data, test_data])

combinedata.info()
combinedata.head(10)

# add a new feature column as 'FamilySize'

combinedata['FamilySize'] = combinedata['Parch'] + combinedata['SibSp'] + 1

# add a new feature column as 'IsAlone'

combinedata['IsAlone'] = 0

combinedata.loc[combinedata['FamilySize'] == 1, 'IsAlone'] = 1

# get name title

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""



combinedata['Title'] = combinedata['Name'].apply(get_title)



# Group all non-common value into one single grouping 'Rare'

combinedata['Title'] = combinedata['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 

                                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')



combinedata['Title'] = combinedata['Title'].replace('Mlle', 'Miss')

combinedata['Title'] = combinedata['Title'].replace('Ms', 'Miss')

combinedata['Title'] = combinedata['Title'].replace('Mme', 'Mrs')





# maping sex into integer value

combinedata['Sex'] = combinedata['Sex'].map( {'female':0, 'male':1} ).astype(int)



# mapping title into integer value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

combinedata['Title'] = combinedata['Title'].map(title_mapping).astype(int)

combinedata['Title'] = combinedata['Title'].fillna(0)



# mapping embarked into integer value

combinedata['Embarked'] = combinedata['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)

# mapping Fare

combinedata.loc[combinedata['Fare'] <= 7.91, 'Fare']  = 0

combinedata.loc[(combinedata['Fare'] > 7.91)&(combinedata['Fare']<=14.454), 'Fare']  = 1

combinedata.loc[(combinedata['Fare'] > 14.454)&(combinedata['Fare']<=31), 'Fare']  = 2

combinedata.loc[(combinedata['Fare'] > 31), 'Fare']  = 3



# mapping Age

combinedata.loc[combinedata['Age'] <= 16, 'Age']  = 0

combinedata.loc[(combinedata['Age'] > 16)&(combinedata['Age']<=32), 'Age']  = 1

combinedata.loc[(combinedata['Age'] > 32)&(combinedata['Age']<=48), 'Age']  = 2

combinedata.loc[(combinedata['Age'] > 48)&(combinedata['Age']<=64), 'Age']  = 3

combinedata.loc[(combinedata['Age'] > 64), 'Age']  = 4



#Feature selection

drop_elements = ['PassengerId', 'Name','SibSp', 'Cabin','Ticket']

combinedata =  combinedata.drop(drop_elements, axis=1)

combinedata.info()
combinedata['Survived'].value_counts().plot.pie(autopct='%1.2ff%%')

feature_iv = {}
combinedata[['Sex','Survived']].groupby(['Sex']).mean().plot.bar() # 0 for female ,1 for male

sex_woe = cacl_woe(combinedata,'Sex','Survived')

sex_iv = calc_iv(sex_woe)

feature_iv['sex_iv'] = sex_iv

print(sex_iv)
combinedata[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

pclass_woe = cacl_woe(combinedata,'Pclass','Survived')

pclass_iv = calc_iv(pclass_woe)

feature_iv['pclass_iv'] = pclass_iv

print(pclass_iv)
combinedata[['Age','Survived']].groupby(['Age']).mean().plot.bar()

age_woe = cacl_woe(combinedata[0:891],'Age','Survived')

age_iv = calc_iv(age_woe)

feature_iv['age_iv'] = age_iv

print(age_iv)
combinedata[['FamilySize','Survived']].groupby(['FamilySize']).mean().plot.bar()

familysize_woe = cacl_woe(combinedata[0:891],'FamilySize','Survived')

familysize_iv = calc_iv(familysize_woe)

feature_iv['familysize_iv'] = familysize_iv

print(familysize_iv)
combinedata[['Title','Survived']].groupby(['Title']).mean().plot.bar()

title_woe = cacl_woe(combinedata[0:891],'Title','Survived')

title_iv = calc_iv(title_woe)

feature_iv['title_iv'] = title_iv

print(title_iv)
combinedata[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar()

embarked_woe = cacl_woe(combinedata[0:891],'Embarked','Survived')

embarked_iv = calc_iv(embarked_woe)

feature_iv['embarked_iv'] = embarked_iv

print(embarked_iv)
combinedata[['Fare','Survived']].groupby(['Fare']).mean().plot.bar()

fare_woe = cacl_woe(combinedata[0:891],'Fare','Survived')

fare_iv = calc_iv(fare_woe)

feature_iv['fare_iv'] = fare_iv

print(fare_iv)
combinedata[['IsAlone','Survived']].groupby(['IsAlone']).mean().plot.bar() # familysize was less than 1,the value was 1

isalone_woe = cacl_woe(combinedata[0:891],'IsAlone','Survived')

isalone_iv = calc_iv(isalone_woe)

feature_iv['isalone_iv'] = isalone_iv

print(isalone_iv)
combinedata[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()

parch_woe = cacl_woe(combinedata[0:891],'Parch','Survived')

parch_iv = calc_iv(parch_woe)

feature_iv['parch_iv'] = parch_iv

print(parch_iv)
sorted(feature_iv.keys())
x_train = combinedata[:891][['Age','Embarked','Fare','Parch','Pclass','Sex','FamilySize','IsAlone','Title']]

y_train = combinedata[:891][['Survived']]

x_test = combinedata[891:].drop(['Survived'],axis=1)

x_train.info()

y_train.info()

x_test.info()
gbm = xgb.XGBClassifier(

n_estimators = 2000,

max_depth=4,

min_child_weight=2,

gamma=0.9,

subsample=0.8,

colsample_bytree=0.8,

objective='binary:logistic',

nthread=-1,

scale_pos_weight=1).fit(x_train,y_train)

predictions = gbm.predict(x_test).astype(int)
StackingSubmission = pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})

StackingSubmission.to_csv("StackingSubmission.csv",index=False)

print('Success')