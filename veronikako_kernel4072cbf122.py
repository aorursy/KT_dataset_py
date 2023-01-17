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
import warnings



warnings.filterwarnings('ignore')
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'



df = pd.read_csv(data)
df.drop(['RISK_MM'], axis=1, inplace=True)
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day

df.drop('Date', axis=1, inplace = True)

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)



X = df.drop(['RainTomorrow', 'Location'], axis=1)

y = df['RainTomorrow']

categorical = [var for var in X.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape

X_train[categorical].isnull().mean()

for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, (X_train[col].isnull().mean()))

for df2 in [X_train, X_test]:

    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)

    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)

    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

X_train[categorical].isnull().sum()

from catboost import CatBoostClassifier, cv

model_cb=CatBoostClassifier(iterations=100, depth=8,

                        learning_rate=0.2,

                        random_seed=42, thread_count=4,

                        rsm=1, 

                        l2_leaf_reg=2,loss_function='MultiClass')

model_cb.fit(X_train, y_train, cat_features=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])
from sklearn.model_selection import cross_val_score,GridSearchCV, validation_curve

parameters = {'depth': [4,6,8,10],

                  'learning_rate' : [0.1,0.2,0.3,0.4],

                  'iterations'    : [20,30, 50, 100]}

grid = GridSearchCV(estimator=model_cb, param_grid = parameters, cv = 2, n_jobs=-1)

grid.fit(X_train, y_train, cat_features=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])    



# Results from Grid Search

print("\n========================================================")

print(" Results from Grid Search " )

print("========================================================")    



print("\n The best estimator across ALL searched params:\n",

         grid.best_estimator_)

    

print("\n The best score across ALL searched params:\n",

          grid.best_score_)

    

print("\n The best parameters across ALL searched params:\n",

          grid.best_params_)
cb_pred=model_cb.predict(X_test)

cb_pred
from sklearn.metrics import classification_report

print(classification_report(y_test, cb_pred))