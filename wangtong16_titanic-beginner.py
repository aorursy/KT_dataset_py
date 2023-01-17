# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')

data_test.head()
data_train.head()
data_train.isna().sum()
data_test.isna().sum()
data_train['Age'].fillna(data_train['Age'].mean(), inplace=True)

data_test['Age'].fillna(data_test['Age'].mean(), inplace=True)

data_train['Embarked'].fillna(data_train['Embarked'].mode()[0], inplace=True) # fill with highest frequency value
data_train.isna().sum()
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace=True)

data_test.isna().sum()
for dataset in [data_train, data_test]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    

    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
data_train.head()
x_used_cols = ['Pclass', 'Sex', 'Fare', 'AgeBin','SibSp', 'FamilySize', 'IsAlone', 'Embarked']

target = 'Survived'

used_cols = [*x_used_cols, target]



data_train = data_train[used_cols]

data_train.describe(include='all')
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



label = OrdinalEncoder()

data_train[['Sex', 'Embarked', 'AgeBin']] =  label.fit_transform(data_train[['Sex', 'Embarked', 'AgeBin']])

data_test[['Sex', 'Embarked', 'AgeBin']] = label.fit_transform(data_test[['Sex', 'Embarked', 'AgeBin']])
data_train.describe()
data_test.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(data_train)
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data_train, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data_train, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data_train, ax = saxis[0,2])



sns.pointplot(x = 'SibSp', y = 'Survived',  data=data_train, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data_train, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data_train, ax = saxis[1,2])
from xgboost import XGBClassifier

from sklearn import model_selection

from sklearn import metrics



xgb = XGBClassifier()

cvs = model_selection.cross_val_score(xgb, data_train[x_used_cols], data_train[target], cv=5)
xgb.fit(data_train[x_used_cols], data_train[target])

result_p = xgb.predict_proba(data_test[x_used_cols])

result_p
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

alg_list = [

    XGBClassifier(),

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

]

alg_results = pd.DataFrame(columns=['S0', 'S1'], data=np.zeros((data_test.shape[0], 2)))

for alg in alg_list:

    alg.fit(data_train[x_used_cols], data_train[target])

    result = alg.predict_proba(data_test[x_used_cols])

    alg_results['S0'] = alg_results['S0'] + result[:,0]

    alg_results['S1'] = alg_results['S1'] + result[:,1]
fresult = np.argmax(alg_results.values, axis = -1)
fresult.shape
pred = pd.DataFrame(columns=['PassengerId', 'Survived'])

pred['PassengerId'] = data_test['PassengerId']

pred['Survived'] = fresult.astype(int)

pred.to_csv("submission.csv", index=False)