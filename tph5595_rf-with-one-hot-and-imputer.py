# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read data from csv
csv_data = pd.read_csv("../input/train.csv")

# Set what we will predict
y = csv_data['Survived']

# Determine feature vectors
features_num = ['Age', 'Fare']
features_cat = ['Pclass', 'Sex', 'Embarked']

# One-hot encoding for categorical columns
data = csv_data[features_cat + features_num]
one_hot_data = pd.get_dummies(data, columns=features_cat)
# Handle missing values
from sklearn.impute import SimpleImputer

# make copy to avoid changing original data (when Imputing)
new_data = one_hot_data.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
# Explore columns left
print(new_data.columns)
print(new_data.head())
# Split data
train_X, val_X, train_y, val_y = train_test_split(new_data, y,random_state = 0)
# define mae calculation
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1, n_estimators=400)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
pred_X = rf_model.predict(val_X)
predications = pred_X.round(0)
rf_val_mae = mean_absolute_error(predications,val_y)

print(rf_val_mae)