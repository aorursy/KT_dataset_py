# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.getcwd()
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
import math, time, random, datetime
# import catboost

from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier, Pool, cv

# Let's be rebels and ignore warnings for now
import warnings


warnings.filterwarnings('ignore')
%matplotlib inline
data_short_test = data_test[['Pclass', 'Fare', 'Age', 'Sex']].copy()
data_short_test['Sex'] = pd.get_dummies(data_short_test['Sex'], prefix='sex')
data_short_test.copy().isna().sum()
from scipy.stats.stats import pearsonr
from sklearn import tree

# /kaggle/input/titanic/test.csv
# /kaggle/input/titanic/train.csv
data_path_test = r'../input/titanic/test.csv'
data_path_train = r'../input/titanic/train.csv'
data = pd.read_csv(data_path_train, index_col='PassengerId')
data_test = pd.read_csv(data_path_test, index_col='PassengerId')
data.shape
data.head(2)
data.info()
data.describe()
data.dtypes
data.isna()
data
# Split the dataframe into data and labels
# X_train = selected_df.drop('Survived', axis=1) # data
# y_train = selected_df.Survived # labels
data_short = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].copy()
data_short = data_short.dropna()
data_short_x = data_short[['Pclass', 'Fare', 'Age', 'Sex']]
data_short_y = data_short['Survived']
data_short_x.shape, data_short_y.shape
data_short_x['Sex'] = pd.get_dummies(data_short_x['Sex'], prefix='sex')
data_short_x
# data_short_x, data_short_y
from sklearn import tree
model_my_learn_ = tree.DecisionTreeClassifier(random_state=241)
# model_my_learn.fit(data_short, data_y)
from sklearn.model_selection import KFold
k_df_fold = KFold(n_splits=10, shuffle=True, random_state=42)
from sklearn.model_selection import cross_val_score
results = cross_val_score(model_my_learn_, data_short_x, data_short_y, cv=k_df_fold, scoring='accuracy')
results
results.mean()
model_my_learn_.fit(data_short_x, data_short_y)
# from sklearn import preprocessing
# data_x_scaled = preprocessing.scale(data_short_x)
# from sklearn.model_selection import cross_val_score
# model_my_learn_ = tree.DecisionTreeClassifier(random_state=241)
# results = cross_val_score(model_my_learn_, data_x_scaled, data_short_y, cv=k_df_fold, scoring='accuracy')
# results
# results = cross_val_score(model_my_learn_, data_short_x, data_short_y, cv=10, scoring='accuracy')
# results
data_test.head()
data_short_test = data_test[['Pclass', 'Fare', 'Age', 'Sex']].copy()
data_short_x_test = data_short_test[['Pclass', 'Fare', 'Age', 'Sex']]

data_short_x_test.isna().sum()
data_short_x_test['Sex'] = pd.get_dummies(data_short_x_test['Sex'], prefix='sex')
data_short_x_test['Age'] = data_short_x_test['Age'].fillna(data_short_x_test['Age'].median())
data_short_x_test['Fare'] = data_short_x_test['Fare'].fillna(data_short_x_test['Fare'].mean())
data_short_x_test.isna().sum()
data_short_x_test.shape
solve_survided = model_my_learn_.predict(data_short_x_test)
ids = data_short_x_test.index
solve_out_table = pd.DataFrame({'PassengerId': ids, 'Survived': solve_survided})
# data_short_x_test['PassengerId'] = data_short_x_test.index
# data_short_x_test['Survived'] = solve_survided
solve_out_table
# predictions = rfc.predict(test_data.drop(['PassengerId'], axis=1))
#set the output as a dataframe and convert to csv file named submission.csv
#output = pd.DataFrame({'PassengerId' : ids, 'Survived': predictions })
solve_out_table.to_csv('submission.csv', index=False)
solve_out_table.to_csv( 'titanic_pred.csv' , index = False )
!ls
