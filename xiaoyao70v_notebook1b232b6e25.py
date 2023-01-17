# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;
train =pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']
full_data = [train, test]
train['Name_length']=train.Name.apply(len)
test['Name_length']=test["Name"].apply(len)
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x)==float else 1)
test['Has_Cabin']=test["Cabin"].apply(lambda x: 0 if pd.isnull(x) else 1)
for dataset in full_data:

    dataset['FamilySzie']=dataset['SibSp']+dataset["Parch"]+1
for dataset in full_data:

    dataset['IsAlone'] =dataset['FamilySzie'].apply(lambda x:1 if x ==1 else 0)
for dataset in full_data:

    dataset['Embarked']=dataset['Embarked'].fillna('S')
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median)
train.head()
train['CategoricalFare'] = pd.qcut(train['Fare'],4)
train.head()
train.groupby('CategoricalFare').count()
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().count()

    age_random_list = np.random.randint(age_avg-age_std,age_avg+age_std,size = age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_random_list

    dataset['Age'] = dataset['Age'].astype(int)
train.head()
train['CategoricalAge'] = pd.cut(train['Age'],5)
train.groupby('CategoricalAge').count()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""
get_title(" sh. Owen Harris")
for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)
train.head()
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map({'female':0,'male':1}).astype(int)

    title_maping ={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] =dataset['Title'].map(title_maping)

    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    dataset.loc[dataset['Fare']<=7.91,'Fare'] = 0

    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare'] = 1

    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] =dataset['Fare'].astype(int)
for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
for dataset in full_data:

    dataset.loc[dataset['Age']<=16,'Age'] = 0

    dataset.loc[(dataset['Age']>16) &(dataset['Age']<=32),'Age'] = 1

    dataset.loc[(dataset['Age']>32) &(dataset['Age']<=48),'Age'] = 2

    dataset.loc[(dataset['Age']>48) &(dataset['Age']<=64),'Age'] = 3

    dataset.loc[dataset['Age']>64,'Age'] = 4
train.head()
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train= train.drop(drop_elements,axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'],axis=1)

test  = test.drop(drop_elements, axis = 1)
train.head()
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features',y=1.05,size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySzie', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])