# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')

input_features = ['Pclass', 'Sex', 'Age']
tr_X = train_dataset[input_features]
tr_Y = train_dataset['Survived']
tt_X = test_dataset[input_features]


tr_X['Sex'] = tr_X['Sex'].replace('female', 0)
tr_X['Sex'] = tr_X['Sex'].replace('male', 1)

tt_X['Sex'] = tt_X['Sex'].replace('female', 0)
tt_X['Sex'] = tt_X['Sex'].replace('male', 1)

#for missing or nan or other null values
imputed_X_train_plus = tr_X.copy()
imputed_X_test_plus = tt_X.copy()

cols_with_missing = (col for col in tr_X.columns 
                                 if tr_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

# train_X, val_X, train_y, val_y = train_test_split(imputed_X_train_plus, tr_Y,random_state = 0)


forest_model = RandomForestRegressor()
forest_model.fit(imputed_X_train_plus, tr_Y)
preds = forest_model.predict(imputed_X_test_plus)

preds = [1 if x > 0.50 else 0 for x in preds]

my_submission = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': preds})
# you could use any filename. We choose submission here
my_submission.to_csv('vinay_submission.csv', index=False)


            





