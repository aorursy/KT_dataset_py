# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data and view a summary

raw_data = pd.read_csv('../input/train.csv')

raw_data.head()
print(raw_data.columns)

print(raw_data.isnull().sum())
raw_data.groupby(['Pclass', 'Sex', 'Embarked'])['Survived'].mean()
# define independent and dependent variables and the features



y = raw_data.Survived

features = ['Pclass', 'Sex', 'Embarked']

X = raw_data[features].copy()
#reformat data to change sex into binary

def reformat(x):

    d_sex = {'female': 0, 'male': 1}

    d_emb = {'C': 0, 'Q': 1, 'S': 2, '3':3}

    x['Sex'] = [d_sex[sex] for sex in x['Sex'].values]

    x['Embarked'] = x['Embarked'].fillna('3')

    x['Embarked'] = [d_emb[elem] for elem in x['Embarked'].values]

    

reformat(X)

X.head()
scaler = StandardScaler()



X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()

model.fit(X_scaled,y)

cross_val_score(model , X_scaled , y , cv=5)
def applyModel(X_test):

    X = X_test[features].copy()

    reformat(X)

    X = scaler.transform(X)

    preds = model.predict(X)

    return preds
X_test = pd.read_csv('../input/test.csv')

preds = applyModel(X_test)

output = pd.DataFrame({'PassengerId': X_test.PassengerId,

                       'Survived': preds})

output.to_csv('submission.csv', index=False)