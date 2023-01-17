# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re as re

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import model_selection, ensemble, linear_model, svm, tree

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_train.shape
data_test.shape
data_train.info()
data_test.info()
passengerid = data_test['PassengerId']
data_train.isnull().sum()
data_test.isnull().sum()
data_full = [data_train, data_test]
data_train[['Pclass','Survived']].groupby('Pclass', as_index = False).mean()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""

for dataset in data_full:

    dataset['Title'] = dataset['Name'].apply(get_title)
pd.crosstab(data_train.Title, data_train.Sex)
pd.crosstab(data_test.Title, data_test.Sex)
for dataset in data_full:

    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Dona','Dr','Rev','Jonkheer','Lady','Major','Mlle','Mme','Ms','Sir'],'Rare')
def simplify_age(age):

    categorical_age = np.zeros(age.count())

    mean_age = age.mean()

    variance_age = age.std()

    na_n = age.isna().sum()

    na_list = np.random.randint(mean_age-variance_age, mean_age+variance_age, na_n)

    age[np.isnan(age)] = na_list

    age = age.astype('int')

    categorical_age = pd.cut(age, bins = [-1,16,30,40,60,100])

    return categorical_age

for dataset in data_full:

    dataset['CategoricalAge']= simplify_age(dataset['Age'])
for dataset in data_full:

    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']

    dataset.loc[dataset.FamilySize == 0, 'Along'] = True

    dataset.loc[dataset.FamilySize != 0,'Along'] = False
for dataset in data_full:

    dataset['Fare']=dataset['Fare'].fillna(0)

    dataset['CategoricalFare']=pd.cut(dataset['Fare'],[-1,7.92, 14.45, 31.0, 600])
for dataset in data_full:

    dataset['Embarked']= dataset['Embarked'].fillna('S')
drop_list = ['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare','Cabin']

data_train = data_train.drop(drop_list,axis=1)

data_test = data_test.drop(drop_list,axis=1)
encoder = LabelEncoder()

Categorical_feature = ['Sex','Embarked','Title','CategoricalAge','Along','CategoricalFare']

for col in Categorical_feature:

    data_train[col] = encoder.fit_transform(data_train[col])

    data_test[col] = encoder.transform(data_test[col])
plt.figure(figsize = (14,10))

sns.heatmap(data_train.corr(), linewidth = 0.1, annot=True)
MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    svm.SVC(probability = True),

    tree.DecisionTreeClassifier(),

    XGBClassifier()

]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.3,random_state = 2)

MLA_columns = ['MLA Name','MLA Parameters','MLA Test Accuracy Mean','MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)

y_train = data_train['Survived']

X_train = data_train.drop('Survived',axis = 1)

row_index = 0

for alg in MLA:

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name']=MLA_name

    MLA_compare.loc[row_index,'MLA Parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv = cv_split)

    MLA_compare.loc[row_index,'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index,'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    row_index+=1
MLA_compare.sort_values(by = 'MLA Test Accuracy Mean', ascending = False,inplace = True)

sns.barplot(x= 'MLA Test Accuracy Mean',y = 'MLA Name', data = MLA_compare)
MLA_compare.loc[MLA_compare['MLA Name']=='SVC','MLA Parameters']
param_list = {'C':np.logspace(-2,2,10)}

my_model_svc = svm.SVC(probability = True)

grid_search_svc = GridSearchCV(my_model_svc, param_grid = param_list, cv = 5)

grid_search_svc.fit(X_train, y_train)
grid_search_svc.best_score_
MLA_compare.loc[MLA_compare['MLA Name']=='AdaBoostClassifier','MLA Parameters']
my_model_ada = ensemble.AdaBoostClassifier()

param_list = {'n_estimators':np.linspace(1,500,10).astype('int'), 'learning_rate':np.logspace(-2,2,5)}

grid_search_ada = GridSearchCV(my_model_ada, param_grid = param_list, cv = 5)

grid_search_ada.fit(X_train, y_train)
grid_search_ada.best_score_
my_model_xgb = XGBClassifier()

MLA_compare.loc[MLA_compare['MLA Name']=='XGBClassifier','MLA Parameters']

param_list = {'n_estimators':np.linspace(1,500,10).astype('int'),'max_depth':np.linspace(1,10,5).astype('int'), 'lambda':np.logspace(-2,2,5)}

grid_search_xgb = GridSearchCV(my_model_xgb, param_grid = param_list, cv = 5)

grid_search_xgb.fit(X_train, y_train)
grid_search_xgb.best_score_
best_model = grid_search_svc.best_estimator_

X_test = data_test

predict=best_model.predict(X_test)

submittion = pd.DataFrame({'PassengerId':passengerid, 'Survived':predict})

submittion.to_csv('Submittion.csv',index = False)