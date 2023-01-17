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
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

X_full_train = pd.read_csv("/kaggle/input/titanic/train.csv",index_col='PassengerId')

X_full_test = pd.read_csv("/kaggle/input/titanic/test.csv")

y = X_full_train.Survived

X_full_train.dropna(axis=0,subset =["Survived"],inplace=True)

X_full_train.drop(['Survived'],axis =1)
X_train_full,X_valid_full,y_train,y_valid = train_test_split(X_full_train, y, train_size=0.8, test_size=0.2,random_state=0)

print(len(X_train_full),len(y_train))

num_features = ['Pclass',"SibSp","Parch"]

cat_features = ['Sex']

all_features = cat_features+num_features



X_train = X_full_train[all_features].copy()

X_valid = X_valid_full[all_features].copy()

X_test = X_full_test[all_features].copy()
from xgboost import XGBClassifier

encoder = LabelEncoder()

X_train['Sex'] = X_train[['Sex']].apply(encoder.fit_transform)

X_valid['Sex'] = X_valid[['Sex']].apply(encoder.transform)

X_test['Sex'] = X_test[['Sex']].apply(encoder.fit_transform)

print(len(X_train),len(y_train))

num_transformer = SimpleImputer()

cat_transform = Pipeline(steps=[

    ('imputer', SimpleImputer()),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, num_features),

        ('cat', cat_transform, cat_features)

    ])



model = XGBClassifier(n_estimators=500,learning_rate = 0.33)

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])
clf.fit(X_train,y)

pred = clf.predict(X_test)

#print(mean_absolute_error(pred,y_valid))

output = pd.DataFrame({'PassengerId': X_full_test.PassengerId, 'Survived': pred})

output.to_csv('my_submission_xgb.csv', index=False)