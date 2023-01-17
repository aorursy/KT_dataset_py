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
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn import preprocessing

# import train.csv

train = pd.read_csv('../input/titanic/train.csv')
train.info()
cols_o = train.select_dtypes(include='object').columns.tolist()

cols_o
cols_i = train.select_dtypes(exclude='object').columns.tolist()

cols_i
train[cols_o]
sel_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']

train[sel_cols].isnull().any()
print(train['Age'])



# SimpleImputer

imp_mean = SimpleImputer(strategy='mean')

imp_mean.fit(train['Age'].values.reshape(-1, 1))

train['Age'] = imp_mean.transform(train['Age'].values.reshape(-1, 1))



print(train['Age'])
print(train[train['Embarked'].isnull()])



# fillna

train['Embarked'] = train['Embarked'].fillna('NA')

print(train[train['Embarked'] == 'NA'])

train[sel_cols].isnull().any()
train[sel_cols]
y = train['Survived']

y
X1 = pd.get_dummies(train[sel_cols])

X1
def model(X, y):

    rf_model = RandomForestClassifier(random_state=0)

    rf_model.fit(X, y)

    y_pred = rf_model.predict(X)

    print('accuracy_score:',metrics.accuracy_score(y_pred, y))

model(X1, y)
X2 = pd.get_dummies(train[sel_cols], drop_first=True)

X2
print(X1.columns)

print(X2.columns)
model(X2, y)
le = preprocessing.LabelEncoder()



for col in ['Sex', 'Embarked']:

    le.fit(train[col])

    le.transform(train[col].values)

    train[col] = le.transform(train[col].values)



X3 = train[sel_cols]

X3
train['Sex'].unique()
train['Embarked'].unique()
model(X3, y)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier



class_models = {

    'SVM': SVC(kernel='linear', C=1.0, random_state=0),

    'LogisticRegression': LogisticRegression(penalty='l2', C=100, random_state=0, max_iter=1000),

    'DecisionTree': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0),

    'knn': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),

    'gnb': GaussianNB(),

    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),

    'ada_boost': AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=0),

    'xgboost': XGBClassifier(max_depth=3, random_state=0),

    'lightgbm': LGBMClassifier(max_depth=3, random_state=0),

}

scores1 = {}



print('-- get_dummies --')

for model_name, model_class in class_models.items():

    model_class.fit(X1, y)

    y_pred = model_class.predict(X1)

    print(model_name, ' accuracy_score:',metrics.accuracy_score(y_pred, y))

    scores1[model_name] = metrics.accuracy_score(y_pred, y)

scores2 = {}



print('-- get_dummies(first_drop) --')

for model_name, model_class in class_models.items():

    model_class.fit(X2, y)

    y_pred = model_class.predict(X2)

    print(model_name, ' accuracy_score:',metrics.accuracy_score(y_pred, y))

    scores2[model_name] = metrics.accuracy_score(y_pred, y)

scores3 = {}



print('-- Label Encoder --')

for model_name, model_class in class_models.items():

    model_class.fit(X3, y)

    y_pred = model_class.predict(X3)

    print(model_name, ' accuracy_score:',metrics.accuracy_score(y_pred, y))

    scores3[model_name] = metrics.accuracy_score(y_pred, y)

scores_df = pd.concat(

    [

        pd.DataFrame(scores1.values(), index=scores1.keys(), columns=['get_dummies']), 

        pd.DataFrame(scores2.values(), index=scores2.keys(), columns=['get_dummies(fd)']),

        pd.DataFrame(scores3.values(), index=scores3.keys(), columns=['Label Encoder'])

    ], axis=1)

scores_df['diff1'] = scores_df['get_dummies'] - scores_df['get_dummies(fd)']

scores_df['diff2'] = scores_df['get_dummies'] - scores_df['Label Encoder']

scores_df